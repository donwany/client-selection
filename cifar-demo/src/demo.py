# import required libraries
from azure.ai.ml import Input
from azure.ai.ml import MLClient
from azure.ai.ml import Output
from azure.ai.ml import command
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute, Environment, Model

subscription_id = "bf1ba48e-b16d-4333-aaec-c4dbb43262bd"
resource_group = "pytorch-dist-demo"
workspace = "pytorch-dist-demo-sys"

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace
)

cluster_name = "cpu-cluster"

cluster = AmlCompute(
        name=cluster_name,
        type="amlcompute",
        size="Standard_NC24",
        location="westus2",
        min_instances=0,
        max_instances=2,
    )
ml_client.begin_create_or_update(cluster)


inputs = {
    "cifar_zip": Input(
        type=AssetTypes.URI_FILE,
        path="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    ),
}

outputs = {
    "cifar": Output(
        type=AssetTypes.URI_FOLDER,
        path=f"azureml://subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace}/datastores/workspaceblobstore/paths/CIFAR-10",
    )
}

job = command(
    code=".",  # local path where the code is stored
    command="python read_write_data.py --input_data ${{inputs.cifar_zip}} --output_folder ${{outputs.cifar}}",
    inputs=inputs,
    outputs=outputs,
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
    compute=cluster_name,
)

if __name__ == '__main__':

    # submit the command
    returned_job = ml_client.jobs.create_or_update(job)
    # get a URL for the status of the job
    returned_job.studio_url

    ml_client.jobs.stream(returned_job.name)

    print(returned_job.name)
    print(returned_job.experiment_name)
    print(returned_job.outputs.cifar)
    print(returned_job.outputs.cifar.path)


