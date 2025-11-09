"""
serve_policy.py
"""

# openVLA imports
import os.path
# ruff: noqa: E402
# import json_numpy
# json_numpy.patch()
import json
import numpy as np
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# openpi imports
import dataclasses
import enum
import logging
import socket
import tyro

from pathlib import Path
import sys
root = Path.home() / "openpi" / "src"
if root.exists():
    sys.path.insert(0, str(root.resolve()))

from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAPolicy(_policy.BasePolicy):
    """Adapter to serve OpenVLA models as an OpenPI policy."""
    def __init__(self, openvla_path: Union[str, Path] = "openvla/openvla-7b", 
                 attn_implementation: Optional[str] = "eager" or "flash_attention_2") -> Path:
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.metadata = None # TODO: may need to update later
        
        logging.info("Loading OpenVLA model %s on %s", openvla_path, self.device)
        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    # TODO: 将action改成np.ndarray
    def infer(self, obs: Dict[str, Any]) -> Dict:
        """
            obs
            {
                "image": np.ndarray, 
                "instruction": str, 
                "unnorm_key": Optional[str]
            }
            
            Returns  {"action": np.ndarray}
        """
        try:
            if double_encode := "encoded" in obs:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(obs.keys()) == 1, "Only uses encoded obs!"
                obs = json.loads(obs["encoded"])

            # Parse obs components
            image, instruction = obs["image"], obs["instruction"]
            unnorm_key = obs.get("unnorm_key", None)
            assert isinstance(image, np.ndarray) and isinstance(instruction, str), "Invalid types for image/instruction!"

            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            # TODO: change the action into np.ndarray
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            return action
        except Exception as e:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return {"error": str(e)}

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"

@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.LIBERO

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8020
    # Record the policy's behavior for debugging.
    record: bool = False
    
    openvla_path: str = "openvla/openvla-7b"

def main(args: Args) -> None:
    policy = OpenVLAPolicy(openvla_path=args.openvla_path, attn_implementation="eager")
    policy_metadata = policy.metadata or None

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")
    # logging
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # create server
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
