"""Creates and runs an Azure ML command job."""

import logging
from pathlib import Path

from azure.ai.ml import MLClient, Output, PyTorchDistribution, command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute, Environment, Model
from azure.identity import DefaultAzureCredential

COMPUTE_NAME = "cluster-distributed-gpu-v4"
# DATA_NAME = "data-fashion-mnist"
# DATA_PATH = Path(Path(__file__).parent.parent, "data")
CONDA_PATH = "/Users/tsiameh/Desktop/paper-writing/FLFSL/FL-Client-Selection/cnn/conda.yml"
CODE_PATH = "/Users/tsiameh/Desktop/paper-writing/FLFSL/FL-Client-Selection/cnn"
MODEL_PATH = "/Users/tsiameh/Desktop/paper-writing/FLFSL/FL-Client-Selection/cnn/"
EXPERIMENT_NAME = "aml_distributed-v4"
MODEL_NAME = "model-distributed-v4"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # credential = DefaultAzureCredential()
    # ml_client = MLClient.from_config(credential=credential)
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id="bf1ba48e-b16d-4333-aaec-c4dbb43262bd",
        resource_group_name="pytorch-dist-demo",
        workspace_name="pytorch-dist-demo-sys"
    )

    # Create the compute cluster.
    # Each Standard_NC24 node has 4 NVIDIA Tesla K80 GPUs.
    cluster = AmlCompute(
        name=COMPUTE_NAME,
        type="amlcompute",
        size="STANDARD_NC6",  # "Standard_NC24",
        location="westus2",
        min_instances=0,
        max_instances=2,
    )
    ml_client.begin_create_or_update(cluster)

    # from azure.ai.ml.entities import AmlCompute
    #
    # gpu_compute_target = "gpu-cluster"
    #
    # try:
    #     # let's see if the compute target already exists
    #     gpu_cluster = ml_client.compute.get(gpu_compute_target)
    #     print(
    #         f"You already have a cluster named {gpu_compute_target}, we'll reuse it as is."
    #     )
    #
    # except Exception:
    #     print("Creating a new gpu compute target...")
    #
    #     gpu_cluster = AmlCompute(
    #         name="gpu-cluster",
    #         type="amlcompute",
    #         size="Standard_NC24",  # 1 x NVIDIA Tesla K80
    #         min_instances=0,
    #         max_instances=4,
    #         idle_time_before_scale_down=180,
    #         #tier="Dedicated",
    #     )
    #
    #     ml_client.begin_create_or_update(gpu_cluster)

    # Create the environment.
    environment = Environment(image="mcr.microsoft.com/azureml/" +
                                    "openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04:latest",
                              conda_file=CONDA_PATH)

    # environment = Environment(image="AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu@latest",
    #                           conda_file=CONDA_PATH)

    # Notice that we specify that we want two nodes/instances, and 4 processes
    # per node/instance.
    # 2 instances * 4 processes per instance = 8 total processes.
    # Azure ML will set the MASTER_ADDR, MASTER_PORT, NODE_RANK, WORLD_SIZE
    # environment variables on each node, in addition to the process-level RANK
    # and LOCAL_RANK environment variables, that are needed for distributed
    # PyTorch training.
    job = command(
        description="Trains a simple neural network on the Fashion-MNIST " +
                    "dataset.",
        experiment_name=EXPERIMENT_NAME,
        compute=COMPUTE_NAME,
        outputs=dict(model=Output(type=AssetTypes.MLFLOW_MODEL)),
        code="./",
        environment="AzureML-pytorch-1.10-ubuntu18.04-py38-cuda11-gpu@latest",
        resources=dict(instance_count=2),
        distribution=dict(type="PyTorch", process_count_per_instance=4),
        command="python train_dnn.py --lr ${{inputs.lr}} --bs ${{inputs.bs}} --localE ${{inputs.localE}} --alpha ${{inputs.alpha}} --dataset ${{inputs.dataset}} --seltype ${{inputs.seltype}} --powd ${{inputs.powd}} --ensize ${{inputs.ensize}} --fracC ${{inputs.fracC}} --size ${{inputs.size}} --save ${{inputs.save}} --optimizer ${{inputs.optimizer}} --model ${{inputs.model}} --rank ${{inputs.rank}} --backend ${{inputs.backend}} --initmethod ${{inputs.initmethod}} --rounds ${{inputs.rounds}} --seed ${{inputs.seed}} --NIID ${{inputs.NIID}} --print_freq ${{inputs.print_freq}}",
        inputs={
            "constantE": True,
            "lr": 0.005,
            "bs": 64,
            "localE": 30,
            "alpha": 2,
            "dataset": "fmnist",
            "seltype": "rand",
            "powd": 2,
            "ensize": 100,
            "fracC": 0.03,
            "size": 3,
            "save": "-p",
            "optimizer": "fedavg",
            "model": "MLP",
            "rank": 0,
            "backend": "nccl",
            "initmethod": "env://",
            "rounds": 300,
            "seed": 2,
            "NIID": False,
            "print_freq": 50
        }
        # + "--model_dir ${{outputs.model}}",
    )
    # job = ml_client.jobs.create_or_update(job)
    # ml_client.jobs.stream(job.name)
    #
    # # Create the model.
    # model_path = f"azureml://jobs/{job.name}/outputs/model"
    # model = Model(name=MODEL_NAME,
    #               path=model_path,
    #               type=AssetTypes.MLFLOW_MODEL)
    # ml_client.models.create_or_update(model)

    import webbrowser

    # submit the job
    returned_job = ml_client.jobs.create_or_update(
        job,
        # Project's name
        experiment_name=EXPERIMENT_NAME,
    )

    # get a URL for the status of the job
    print("The url to see your live job running is returned by the sdk:")
    print(returned_job.studio_url)
    # open the browser with this url
    webbrowser.open(returned_job.studio_url)

    # print the pipeline run id
    print(
        f"The pipeline details can be access programmatically using identifier: {returned_job.name}"
    )
    # saving it for later in this notebook
    small_scale_run_id = returned_job.name


if __name__ == "__main__":
    main()
