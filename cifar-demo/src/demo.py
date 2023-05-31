# import required libraries
from azure.ai.ml import Input
from azure.ai.ml import MLClient
from azure.ai.ml import Output
from azure.ai.ml import command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute

# dependencies_dir = "./env"
# os.makedirs(dependencies_dir, exist_ok=True)

subscription_id = "bf1ba48e-b16d-4333-aaec-c4dbb43262bd"
resource_group = "pytorch-dist-demo"
workspace = "pytorch-dist-demo-sys"

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()


from azure.ai.ml.entities import Environment

custom_env_name = "sklearn-env"

job_env = Environment(
    name=custom_env_name,
    description="Custom environment for sklearn image classification",
    conda_file="/Users/tsiameh/Desktop/paper-writing/FLFSL/FL-Client-Selection/cifar-demo/src/conda.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)

print(
    f"Environment with name {job_env.name} is registered to workspace, the environment version is {job_env.version}"
)

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace
)

job_env = ml_client.environments.create_or_update(job_env)

cluster_name = "cpu-cluster"

cluster = AmlCompute(
        name=cluster_name,
        type="amlcompute",
        size="Standard_NC6",
        location="eastus2",
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
        mode="upload",
        path=f"azureml://datastores/workspaceblobstore/paths/CIFAR-10",
    )
}

job = command(
    code="./",  # local path where the code is stored
    command="python read_write_data.py --input_data ${{inputs.cifar_zip}} --output_folder ${{outputs.cifar}}",
    inputs=inputs,
    outputs=outputs,
    experiment_name="cifar-data-upload",
    # environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1",
    environment=f"{job_env.name}:{job_env.version}",
    compute=cluster_name,
    resources=dict(instance_count=1)
)

if __name__ == '__main__':

    # submit the command
    returned_job = ml_client.jobs.create_or_update(job, experiment_name="cifar-dataset")
    # get a URL for the status of the job
    returned_job.studio_url

    ml_client.jobs.stream(returned_job.name)

    print(returned_job.name)
    print(returned_job.experiment_name)
    print(returned_job.outputs.cifar)
    print(returned_job.outputs.cifar.path)


