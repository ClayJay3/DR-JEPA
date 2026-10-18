"""`python -m drjepa <command> [args]` -- one entry point for every tool.

    gen             generate simulator training episodes
    label-zed       ZED exports (tools/zed) -> labelled real episodes
    train           train the GeoBEV perception network
    eval-perception open-loop perception benchmark on labelled episodes
    eval-drive      closed-loop driving benchmark in the simulator
    export          export a checkpoint to ONNX (+ verify, + CPU latency)

The endless driving demo is ./demo.py at the repo root.

Run `python -m drjepa <command> -h` for each command's options.
"""

import importlib
import sys

COMMANDS = {
    "gen": "drjepa.sim.datagen",
    "label-zed": "drjepa.data.recordings",
    "train": "drjepa.perception.train",
    "eval-perception": "drjepa.eval.perception",
    "eval-drive": "drjepa.eval.closed_loop",
    "export": "drjepa.perception.export",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 1)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"drjepa {cmd}"
    importlib.import_module(COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()
