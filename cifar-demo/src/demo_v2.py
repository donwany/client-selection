from azure.ai.ml import Input
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()

subscription_id = "bf1ba48e-b16d-4333-aaec-c4dbb43262bd"
resource_group = "pytorch-dist-demo"
workspace = "pytorch-dist-demo-sys"

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace
)

gpu_compute_taget = "train-cifar-model-gpu"

# === Note on path ===
# can be can be a local path or a cloud path. AzureML supports https://`, `abfss://`, `wasbs://` and `azureml://` URIs.
# Local paths are automatically uploaded to the default datastore in the cloud.
# More details on supported paths: https://docs.microsoft.com/azure/machine-learning/how-to-read-write-data-v2#supported-paths

# try:
#     # let's see if the compute target already exists
#     gpu_cluster = ml_client.compute.get(gpu_compute_taget)
#     print(
#         f"You already have a cluster named {gpu_compute_taget}, we'll reuse it as is."
#     )
#
# except Exception:
#
#     print("Creating a new gpu compute target...")

# Let's create the Azure ML compute object with the intended parameters
gpu_cluster = AmlCompute(
    # Name assigned to the compute cluster
    name=gpu_compute_taget,
    # Azure ML Compute is the on-demand VM service
    type="amlcompute",
    # VM Family
    size="Standard_NC6",
    # Minimum running nodes when there is no job running
    min_instances=0,
    # Nodes in cluster
    max_instances=2,
    # How many seconds will the node running after the job termination
    idle_time_before_scale_down=180,
    # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
    tier="Dedicated",
)

# Now, we pass the object to MLClient's create_or_update method
gpu_cluster = ml_client.begin_create_or_update(gpu_cluster).result()

print(
    f"AMLCompute with name {gpu_cluster.name} is created, the compute size is {gpu_cluster.size}"
)

inputs = {
    "cifar": Input(
        type=AssetTypes.URI_FOLDER,
        path="azureml://datastores/workspaceblobstore/paths/CIFAR-10"
        # returned_job.outputs.cifar.path
    ),
    # path="azureml:azureml_stoic_cartoon_wgb3lgvgky_output_data_cifar:1"), #path="azureml://datastores/workspaceblobstore/paths/azureml/stoic_cartoon_wgb3lgvgky/cifar/"),
    "epoch": 10,
    "batchsize": 64,
    "workers": 2,
    "lr": 0.01,
    "momen": 0.9,
    "prtfreq": 200,
    "output": "./",
}

job = command(
    code="./",  # local path where the code is stored
    command="python train.py --data-dir ${{inputs.cifar}} --epochs ${{inputs.epoch}} --batch-size ${{inputs.batchsize}} --workers ${{inputs.workers}} --learning-rate ${{inputs.lr}} --momentum ${{inputs.momen}} --print-freq ${{inputs.prtfreq}} --model-dir ${{inputs.output}}",
    inputs=inputs,
    environment="AzureML-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu@latest",
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
    # ml_client.jobs.create_or_update(job)
    import webbrowser

    # submit the job
    returned_job = ml_client.jobs.create_or_update(
        job,
        # Project's name
        # experiment_name=EXPERIMENT_NAME,
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
