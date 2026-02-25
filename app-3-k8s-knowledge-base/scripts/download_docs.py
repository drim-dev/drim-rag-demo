"""Download Kubernetes documentation markdown files from the official repository."""

import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "data" / "k8s-docs"

REPO_TARBALL_URL = (
    "https://github.com/kubernetes/website/archive/refs/heads/main.tar.gz"
)

TARGET_DIRS = [
    "content/en/docs/concepts/",
    "content/en/docs/tasks/",
    "content/en/docs/tutorials/",
]

IMPORTANT_PAGES = {
    "concepts/overview/what-is-kubernetes.md",
    "concepts/overview/components.md",
    "concepts/workloads/pods/_index.md",
    "concepts/workloads/pods/pod-lifecycle.md",
    "concepts/workloads/controllers/deployment.md",
    "concepts/workloads/controllers/statefulset.md",
    "concepts/workloads/controllers/daemonset.md",
    "concepts/workloads/controllers/job.md",
    "concepts/workloads/controllers/replicaset.md",
    "concepts/services-networking/service.md",
    "concepts/services-networking/ingress.md",
    "concepts/services-networking/dns-pod-service.md",
    "concepts/services-networking/network-policies.md",
    "concepts/storage/volumes.md",
    "concepts/storage/persistent-volumes.md",
    "concepts/storage/storage-classes.md",
    "concepts/configuration/configmap.md",
    "concepts/configuration/secret.md",
    "concepts/configuration/manage-resources-containers.md",
    "concepts/scheduling-eviction/kube-scheduler.md",
    "concepts/scheduling-eviction/assign-pod-node.md",
    "concepts/scheduling-eviction/taint-and-toleration.md",
    "concepts/cluster-administration/namespaces.md",
    "concepts/cluster-administration/logging.md",
    "concepts/cluster-administration/manage-deployment.md",
    "concepts/security/rbac-good-practices.md",
    "concepts/security/pod-security-standards.md",
    "concepts/architecture/nodes.md",
    "concepts/architecture/control-plane-node-communication.md",
    "concepts/extend-kubernetes/api-extension/custom-resources.md",
    "concepts/extend-kubernetes/operator.md",
    "concepts/policy/resource-quotas.md",
    "concepts/policy/limit-range.md",
    "tasks/run-application/horizontal-pod-autoscale.md",
    "tasks/run-application/horizontal-pod-autoscale-walkthrough.md",
    "tasks/configure-pod-container/configure-liveness-readiness-startup-probes.md",
    "tasks/configure-pod-container/configure-pod-configmap.md",
    "tasks/configure-pod-container/configure-volume-storage.md",
    "tasks/manage-kubernetes-objects/declarative-config.md",
    "tasks/access-application-cluster/port-forward-access-application-cluster.md",
    "tasks/debug/debug-application/debug-pods.md",
    "tasks/debug/debug-cluster/_index.md",
    "tasks/administer-cluster/safely-drain-node.md",
    "tasks/administer-cluster/manage-resources/memory-default-namespace.md",
    "tutorials/kubernetes-basics/_index.md",
    "tutorials/stateless-application/guestbook.md",
    "tutorials/stateful-application/basic-stateful-set.md",
    "tutorials/security/cluster-level-pss.md",
    "tutorials/services/connect-applications-service.md",
}


def normalize_path(tar_path: str) -> str | None:
    """Extract the docs-relative path from the tarball entry.

    Tarball entries look like: website-main/content/en/docs/concepts/...
    We strip the prefix down to concepts/... for matching against IMPORTANT_PAGES.
    """
    parts = tar_path.split("/")
    try:
        docs_idx = parts.index("docs")
    except ValueError:
        return None
    return "/".join(parts[docs_idx + 1:])


def download_docs():
    print("Downloading Kubernetes website repository tarball...")
    response = requests.get(REPO_TARBALL_URL, stream=True, timeout=120)
    response.raise_for_status()

    content = BytesIO(response.content)
    print(f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    extracted = 0

    with tarfile.open(fileobj=content, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".md"):
                continue

            in_target_dir = any(d in member.name for d in TARGET_DIRS)
            if not in_target_dir:
                continue

            relative = normalize_path(member.name)
            if relative is None:
                continue

            if relative not in IMPORTANT_PAGES:
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            safe_name = relative.replace("/", "__")
            dest = DOCS_DIR / safe_name
            dest.write_bytes(f.read())
            extracted += 1
            print(f"  [{extracted}] {relative}")

    print(f"\nExtracted {extracted} documentation files to {DOCS_DIR}")


if __name__ == "__main__":
    download_docs()
