import json


if __name__ == "__main__":
    discovery = {
        "GPU": 8,
        "GPU_Name": ["v100-sxm2", "v100-pcie"],
        "Time": 8,
        "Multi-GPU": True,
        "Slurm": True,
        "Models": {
            "Alpaca-13B": ["models/alpaca-13b", 4, "fp16"],
            "T0++": ["bigscience/T0pp", 2, "fp32"],
            "Flan-T5-XXL": ["google/flan-t5-xxl", 8, "fp16"],
            "Alpaca-7B": ["models/alpaca-7b", 4, "fp16"],
            "T0-3B": ["bigscience/T0_3B", 8, "fp32"],
            "Flan-T5-XL": ["google/flan-t5-xl", 32, "fp16"],
            "Flan-T5-Large": ["google/flan-t5-large", 32, "fp16"],
            "Flan-T5-Base": ["google/flan-t5-base", 128, "fp16"],
            "Flan-T5-Base-PrefixLayer": ["models/flan-t5-base-layer", 128, "fp16"],
            "Flan-T5-Base-PrefixEmbed": ["models/flan-t5-base-embed", 128, "fp16"],
            "Flan-T5-Small": ["google/flan-t5-small", 128, "fp16"],
        }
    }

    frink = {
        "GPU": 1,
        "GPU_Name": ["RTX-8000"],
        "Time": 72,
        "Multi-GPU": False,
        "Slurm": True,
        "Models": {
            "Alpaca-13B": ["models/alpaca-13b", 4, "fp16"],
            "T0++": ["bigscience/T0pp", 4, "fp32"],
            "Flan-T5-XXL": ["google/flan-t5-xxl", 8, "fp16"],
            "Alpaca-7B": ["models/alpaca-7b", 16, "fp16"],
            "T0-3B": ["bigscience/T0_3B", 16, "fp32"],
            "Flan-T5-XL": ["google/flan-t5-xl", 32, "fp16"],
            "Flan-T5-Large": ["google/flan-t5-large", 64, "fp16"],
            "Flan-T5-Base": ["google/flan-t5-base", 128, "fp16"],
            "Flan-T5-Base-PrefixLayer": ["models/flan-t5-base-layer", 128, "fp16"],
            "Flan-T5-Base-PrefixEmbed": ["models/flan-t5-base-embed", 128, "fp16"],
            "Flan-T5-Small": ["google/flan-t5-small", 256, "fp16"],
        }
    }

    fluidstack = {
        "GPU": 8,
        "GPU_Name": ["a100-sxm4"],
        "Time": None,
        "Multi-GPU": True,
        "Slurm": False,
        "Models": {
            "Alpaca-13B": ["models/alpaca-13b", 8, "bf16"],
            "T0++": ["bigscience/T0pp", 8, "fp32"],
            "Flan-T5-XXL": ["google/flan-t5-xxl", 16, "bf16"],
            "Alpaca-7B": ["models/alpaca-7b", 16, "bf16"],
            "T0-3B": ["bigscience/T0_3B", 16, "fp32"],
            "Flan-T5-XL": ["google/flan-t5-xl", 32, "bf16"],
            "Flan-T5-Large": ["google/flan-t5-large", 64, "bf16"],
            "Flan-T5-Base": ["google/flan-t5-base", 128, "bf16"],
            "Flan-T5-Base-PrefixLayer": ["models/flan-t5-base-layer", 128, "bf16"],
            "Flan-T5-Base-PrefixEmbed": ["models/flan-t5-base-embed", 128, "bf16"],
            "Flan-T5-Small": ["google/flan-t5-small", 256, "bf16"],
        }
    }

    json.dump(discovery, open("./discovery.json", "w"))
    json.dump(frink, open("./frink.json", "w"))
    json.dump(fluidstack, open("./fluid-a100.json", "w"))


