import boto3, sagemaker, subprocess, os, tarfile
from sagemaker import get_execution_role
from sagemaker.model import Model

role = get_execution_role()
sess = sagemaker.Session()
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
repo_name = "sam2-direct-github"

def build_and_push_image():
    ecr = boto3.client('ecr')
    try:
        ecr.create_repository(repositoryName=repo_name)
    except ecr.exceptions.RepositoryAlreadyExistsException:
        pass
    uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest"
    subprocess.run(f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account}.dkr.ecr.{region}.amazonaws.com", shell=True)
    subprocess.run(["docker", "build", "-t", repo_name, "."], check=True)
    subprocess.run(["docker", "tag", repo_name, uri], check=True)
    subprocess.run(["docker", "push", uri], check=True)
    return uri

def package_empty_model():
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("config", arcname="config")
    return sess.upload_data("model.tar.gz", key_prefix="sam2_artifacts")

def deploy_model(image_uri, model_data):
    model = Model(
        image_uri=image_uri,
        role=role,
        model_data=model_data,
        name="sam2-direct-github",
        entry_point="inference.py"
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.2xlarge",
        endpoint_name="sam2-direct-endpoint"
    )
    print(f"Endpoint deployed: {predictor.endpoint_name}")
    return predictor

if __name__ == "__main__":
    image_uri = build_and_push_image()
    model_data = package_empty_model()
    deploy_model(image_uri, model_data)

