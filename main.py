from experiment import Experiment


if __name__ == "__main__":
    experiment = Experiment("google/flan-t5-xl", devices=[0], precision="bf16")
    experiment.add_tasks_by_name(
        task_name="Conceptual_Combinations_Adversarial",
        output_dir="./test/",
        batch_size="8",
        instruction="BBL/Unobserved/4",
        shot_count="0",
        eval_by_logit=True,
    )
    experiment.inference()
