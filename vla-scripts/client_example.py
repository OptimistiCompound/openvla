import requests
import json_numpy
json_numpy.patch()
import numpy as np

def main():
    action = requests.post(
        "http://0.0.0.0:8004/act",
        json={"image": np.zeros((256, 256, 3), dtype=np.uint8).tolist(),
              "instruction": "do something"}
    ).json()
    print("Received action:", action)

if __name__ == "__main__":
    main()
