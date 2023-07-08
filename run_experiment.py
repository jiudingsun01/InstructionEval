import json
import os
import pandas as pd
from experiment import Experiment, DATASET2CONFIGS
from torch.cuda import empty_cache
import time


class Task:

    _ROOT_DIR = "./results"

    def __init__(self, row: pd.Series, model_config):
        self.shot_count = str(row["Shots"])
        self.input_dir, self.config_dir = DATASET2CONFIGS[row["Dataset"]]
        self.output_dir = os.path.join(self._ROOT_DIR, "{}_{}S".format(row["Model"], self.shot_count), row["Dataset"],
                                       "{}_{}".format(row["Collection"], row["Type"]), str(row["ID"]))
        _, self.batch_size, _ = model_config[row["Model"]]
        self.instr = "{}/{}/{}".format(row["Collection"], row["Type"], str(row["ID"]))
        self.eval_by_logit = True if row["Metric"] == "Logits" else False

    def get_input_dict(self):
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "config_dir": self.config_dir,
            "batch_size": self.batch_size,
            "instruction": self.instr,
            "shot_count": self.shot_count,
            "eval_by_logit": self.eval_by_logit
        }


class TasksQueue:

    def __init__(self, model, gpus, precision, gpu_name=None, hours=8):
        self.tasks = []
        self.model = model
        self.gpus = gpus
        self.gpu_name = gpu_name
        self.precision = precision
        self.hours = hours

        self.json_dir = None

    def is_empty(self):
        return len(self.tasks) == 0

    def add_task(self, task: Task):
        self.tasks.append(task)

    def generate_tasks_json(self, json_dir: str):
        json_items = [task.get_input_dict() for task in self.tasks]
        with open(json_dir, "w") as f:
            json.dump(json_items, f)
        f.close()
        self.json_dir = json_dir

    def generate_tasks_slurm(self, script_dir):
        assert self.json_dir is not None
        with open(script_dir, "w") as f:
            f.write("#!/bin/bash" + "\n")
            if len(self.gpus) > 1:
                f.write("#SBATCH --partition=multigpu" + "\n")
                f.write("#SBATCH --nodes=1" + "\n")
                f.write("#SBATCH --gres=gpu:{}:{}".format(self.gpu_name, len(self.gpus)) + "\n")
                f.write("#SBATCH --ntasks-per-node={}".format(len(self.gpus)) + "\n")
                f.write("#SBATCH --mem=128GB" + "\n")
                f.write("#SBATCH --time={}:00:00".format(self.hours) + "\n")
                f.write("\n")
                f.write("source /work/frink/sun.jiu/miniconda3/bin/activate" + "\n")
                f.write("cd /work/frink/sun.jiu/FlanRobustness" + "\n")
                f.write("conda activate pytorch_env" + "\n")
                command = "python experiment.py --model_name_or_path \"{}\" --tasks_dir \"{}\" --devices".format(
                    self.model, self.json_dir,
                )
                for gpu_id in self.gpus:
                    command += " " + str(gpu_id)
                f.write(command + "\n")
            else:
                f.write("#SBATCH --partition=gpu" + "\n")
                f.write("#SBATCH --nodes=1" + "\n")
                f.write("#SBATCH --gres=gpu:{}:1".format(self.gpu_name) + "\n")
                f.write("#SBATCH --ntasks=1" + "\n")
                f.write("#SBATCH --mem=64GB" + "\n")
                f.write("#SBATCH --time={}:00:00".format(self.hours) + "\n")
                f.write("\n")
                f.write("source /work/frink/sun.jiu/miniconda3/bin/activate" + "\n")
                f.write("cd /work/frink/sun.jiu/FlanRobustness" + "\n")
                f.write("conda activate pytorch_env" + "\n")
                command = "python experiment.py --model_name_or_path \"{}\" --tasks_dir \"{}\" --devices".format(
                    self.model, self.json_dir,
                )
                for gpu_id in self.gpus:
                    command += " " + str(gpu_id)
                f.write(command + "\n")
            f.close()

    def run(self):
        exp = Experiment(model_name_or_path=self.model, devices=self.gpus, precision=self.precision)
        for task in self.tasks:
            exp.add_tasks(**task.get_input_dict())
        exp.inference()


class Generator:

    def __init__(self, tar_dir):
        self.tar_dir = tar_dir
        self.df = pd.DataFrame(columns=["Model", "Dataset", "Collection", "Type", "ID", "Shots", "Performance", "Metric", "KL", "EncoderL2", "Definition", "Include?"])

    def generate_paraphase_experiment_csv(self, paraphrases_dir):
        task_ids = [f for f in os.listdir(paraphrases_dir) if "DS" not in f]
        for task_id in task_ids:
            task_dir = os.path.join(paraphrases_dir, task_id)
            prompt_files = [f for f in os.listdir(task_dir) if "stat" not in f]

            for prompt_file in prompt_files:
                prompt_id = prompt_file.replace(".json", "")
                prompt_file_dir = os.path.join(task_dir, prompt_file)
                prompt = json.load(open(prompt_file_dir, "r"))
                try:
                    items = prompt["choices"]
                except KeyError:
                    continue
                prompt_stat_file_dir = os.path.join(task_dir, prompt_id + "_stat.json")
                prompt_stat = json.load(open(prompt_stat_file_dir, "r"))
                for instruction_id in range(len(items)):
                    definition = items[instruction_id]["message"]["content"]
                    KL = prompt_stat[instruction_id]["KL_Divergence"]
                    EncoderL2 = prompt_stat[instruction_id]["Encoder_L2"]
                    self.df.loc[len(self.df.index)] = ["Flan-T5-XL", "Language_Identification_Paraphrased", task_id, prompt_id, instruction_id, 0, "PD", "Logits", KL, EncoderL2, definition, True]
        if not os.path.exists(self.tar_dir):
                os.makedirs(self.tar_dir, exist_ok=True)
        tar_file = os.path.join(self.tar_dir, "paraphrases.csv")
        self.df.to_csv(tar_file, index=False)


class Dispatcher:

    _SLURM_FOLDER = "./sbatches"
    _JSON_FOLDER = "./task_jsons"
    _DATA_FOLDER = "./results"

    def __init__(self, data_dir, server="discovery"):
        self.data_dir = data_dir
        self.df = pd.read_csv(data_dir, index_col=None)
        self.server_configs = json.load(open(os.path.join("./servers", "{}.json".format(server)), "r"))
        if not os.path.exists(self._SLURM_FOLDER):
            os.makedirs(self._SLURM_FOLDER, exist_ok=True)
        if not os.path.exists(self._JSON_FOLDER):
            os.makedirs(self._JSON_FOLDER, exist_ok=True)
    
    def batch_size_reduced_by_half(self):
        for key in self.server_configs["Models"].keys():
            self.server_configs["Models"][key][1] = int(self.server_configs["Models"][key][1] / 2)

    def run(self, gpus=[0], idxs=None):
        tasks = dict()
        idxs = self.df.index if idxs is None else idxs
        for i in self.df.index:
            if self.df.loc[i]["Performance"] == "PD" and i in idxs:
                model_name_or_path, batch_size, precision = self.server_configs["Models"][self.df.loc[i]["Model"]]
                if model_name_or_path not in tasks.keys():
                    tasks[model_name_or_path] = TasksQueue(model=model_name_or_path, gpus=gpus, precision=precision)
                tasks[model_name_or_path].add_task(Task(self.df.loc[i], self.server_configs["Models"]))

        for key in tasks.keys():
            queue = tasks[key]
            queue.run()
            empty_cache()

    def generate_tasks_files(self, n_files):
        tasks = dict()
        count = 0
        for i in self.df.index:
            if self.df.loc[i]["Performance"] == "PD":
                model_name_or_path, batch_size, precision = self.server_configs["Models"][self.df.loc[i]["Model"]]
                if model_name_or_path not in tasks.keys():
                    tasks[model_name_or_path] = [
                        TasksQueue(
                            model=model_name_or_path,
                            gpus=[0],
                            precision=precision,
                        ) for j in range(n_files)]
                tasks[model_name_or_path][count % n_files].add_task(Task(self.df.loc[i], self.server_configs["Models"]))
                count += 1

        for model in tasks.keys():
            _, file_name = model.split("/")
            for i in range(len(tasks[model])):
                queue = tasks[model][i]
                if not queue.is_empty():
                    json_idx = 0
                    while f"{file_name}_tasks{json_idx}.json" in os.listdir("./"):
                        json_idx += 1
                    queue.generate_tasks_json(f"./{file_name}_tasks{json_idx}.json")
                    print("New JSON: "+ f"{file_name}_tasks{json_idx}.json")

    def slurm_dispatch(self, n_batch=None):
        tasks = dict()
        assert self.server_configs["Slurm"]
        n_batch = self.server_configs["GPU"] if n_batch is None else n_batch
        gpu_names = self.server_configs["GPU_Name"]
        sbatch_list = []
        for i in self.df.index:

            if self.df.loc[i]["Performance"] == "PD":
                model_name_or_path, batch_size, precision = self.server_configs["Models"][self.df.loc[i]["Model"]]
                if model_name_or_path not in tasks.keys():
                    tasks[model_name_or_path] = [
                        TasksQueue(
                            model=model_name_or_path,
                            gpus=[0],
                            precision=precision,
                            gpu_name=gpu_names[j % len(gpu_names)],
                            hours=self.server_configs["Time"]
                        ) for j in range(n_batch)
                    ]
                tasks[model_name_or_path][i % n_batch].add_task(Task(self.df.loc[i], self.server_configs["Models"]))

        file_name = time.strftime("%m%d", time.localtime())
        for key in tasks.keys():
            for i in range(len(tasks[key])):
                queue = tasks[key][i]
                if not queue.is_empty():
                    json_idx = 0

                    while f"{file_name}_{json_idx}.json" in os.listdir(self._JSON_FOLDER):
                        json_idx += 1
                    queue.generate_tasks_json(os.path.join(self._JSON_FOLDER, f"{file_name}_{json_idx}.json"))

                    slurm_idx = 0
                    while f"{file_name}_{slurm_idx}.sh" in os.listdir(self._SLURM_FOLDER):
                        slurm_idx += 1

                    queue.generate_tasks_slurm(os.path.join(self._SLURM_FOLDER, f"{file_name}_{slurm_idx}.sh"))
                    sbatch_list.append(os.path.join(self._SLURM_FOLDER, f"{file_name}_{slurm_idx}.sh"))

        for sbatch_file in sbatch_list:
            os.system("sbatch {}".format(sbatch_file))

    def gather(self):
        new_result = 0
        for i in self.df.index:
            if self.df.loc[i]["Performance"] in ["IP", "PD", "W"]:
                path1 = self.df.loc[i]["Model"]
                path1 += "_{}S".format(self.df.loc[i]["Shots"])
                path2 = self.df.loc[i]["Dataset"]
                path3 = str(self.df.loc[i]["Collection"]) + "_" + str(self.df.loc[i]["Type"])
                path4 = str(self.df.loc[i]["ID"])

                info_path = os.path.join(self._DATA_FOLDER, path1, path2, path3, path4, "info.json")
                if os.path.exists(info_path):
                    info = json.load(open(info_path, "r"))
                    self.df.loc[i, "Performance"] = info["performance"]
                    new_result += 1

        self.df = self.df.sort_values(by=["Model", "Dataset", "Collection", "Type", "Shots", "ID", "Metric"])
        self.df = self.df.drop_duplicates()
        print("New results: {}".format(new_result))
        self.df.to_csv(self.data_dir, index=False)

    def add_model_config(self, model, model_name_or_path, batch_size, precision):
        self.server_configs["Models"][model] = [model_name_or_path, batch_size, precision]

    def align_task_schedule(self, target_dir):
        tar_df = pd.read_csv(target_dir, index_col=None)
        model_name = tar_df.loc[0]["Model"]

        for i in self.df.index:
            dataset = self.df.loc[i]["Dataset"]
            collection = self.df.loc[i]["Collection"]
            instruction_type = self.df.loc[i]["Type"]
            idx = self.df.loc[i]["ID"]
            shots = self.df.loc[i]["Shots"]
            metric = self.df.loc[i]["Metric"]
            include = self.df.loc[i]["Include?"]

            same_data = tar_df.loc[
                (tar_df["Dataset"] == dataset) & (tar_df["Collection"] == collection) &
                (tar_df["Type"] == instruction_type) & (tar_df["ID"] == idx) &
                (tar_df["Shots"] == shots) & (tar_df["Metric"] == metric)]

            if not len(same_data):
                tar_df.loc[len(tar_df.index)] = [model_name, dataset, collection, instruction_type, idx, shots, "TBD", metric, include]

        tar_df = tar_df.sort_values(by=["Model", "Dataset", "Collection", "Type", "Shots", "ID", "Metric"])
        tar_df.to_csv(target_dir, index=False)


if __name__ == "__main__":

    dispatcher = Dispatcher("./result_csv/Adversial/Logical_Sequence.csv", "fluid-a100")
    dispatcher.run()
    dispatcher.gather()
    # dispatcher.run([0])


    




