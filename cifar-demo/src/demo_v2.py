from azure.ai.ml import command
from azure.ai.ml.entities import Data
from azure.ai.ml import Input
from azure.ai.ml import Output
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# === Note on path ===
# can be can be a local path or a cloud path. AzureML supports https://`, `abfss://`, `wasbs://` and `azureml://` URIs.
# Local paths are automatically uploaded to the default datastore in the cloud.
# More details on supported paths: https://docs.microsoft.com/azure/machine-learning/how-to-read-write-data-v2#supported-paths

subscription_id = "bf1ba48e-b16d-4333-aaec-c4dbb43262bd"
resource_group = "pytorch-dist-demo"
workspace = "pytorch-dist-demo-sys"

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace
)


inputs = {
    "cifar": Input(
        type=AssetTypes.URI_FOLDER, path=returned_job.outputs.cifar.path
    ),  # path="azureml:azureml_stoic_cartoon_wgb3lgvgky_output_data_cifar:1"), #path="azureml://datastores/workspaceblobstore/paths/azureml/stoic_cartoon_wgb3lgvgky/cifar/"),
    "epoch": 10,
    "batchsize": 64,
    "workers": 2,
    "lr": 0.01,
    "momen": 0.9,
    "prtfreq": 200,
    "output": "./outputs",
}

job = command(
    code="./src",  # local path where the code is stored
    command="python train.py --data-dir ${{inputs.cifar}} --epochs ${{inputs.epoch}} --batch-size ${{inputs.batchsize}} --workers ${{inputs.workers}} --learning-rate ${{inputs.lr}} --momentum ${{inputs.momen}} --print-freq ${{inputs.prtfreq}} --model-dir ${{inputs.output}}",
    inputs=inputs,
    environment="azureml:AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu:6",
    compute="gpu-cluster",  # Change the name to the gpu cluster of your workspace.
    instance_count=2,  # In this, only 2 node cluster was created.
    distribution={
        "type": "PyTorch",
        # set process count to the number of gpus per node
        # NV6 has only 1 GPU
        "process_count_per_instance": 1,
    },
)

if __name__ == '__main__':
    # submit the job
    ml_client.jobs.create_or_update(job)