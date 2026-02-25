"""Download selected Go source files from the Kubernetes repository."""

from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent
CODE_DIR = BASE_DIR / "data" / "k8s-code"

RAW_URL = "https://raw.githubusercontent.com/kubernetes/kubernetes/master"

IMPORTANT_FILES = [
    "pkg/apis/core/types.go",
    "pkg/apis/apps/types.go",
    "pkg/apis/batch/types.go",
    "pkg/apis/networking/types.go",
    "pkg/apis/rbac/types.go",
    "pkg/apis/storage/types.go",
    "pkg/apis/autoscaling/types.go",
    "pkg/scheduler/schedule_one.go",
    "pkg/scheduler/scheduler.go",
    "pkg/controller/deployment/deployment_controller.go",
    "pkg/controller/replicaset/replica_set.go",
    "pkg/controller/statefulset/stateful_set.go",
    "pkg/controller/daemon/daemon_controller.go",
    "pkg/controller/job/job_controller.go",
    "pkg/controller/namespace/namespace_controller.go",
    "pkg/kubelet/kubelet.go",
    "pkg/kubelet/pod_workers.go",
    "pkg/proxy/iptables/proxier.go",
    "cmd/kubectl/kubectl.go",
    "cmd/kube-apiserver/apiserver.go",
    "cmd/kube-scheduler/scheduler.go",
    "cmd/kube-controller-manager/controller-manager.go",
    "pkg/controller/endpoint/endpoints_controller.go",
    "pkg/controller/volume/persistentvolume/pv_controller.go",
    "pkg/controller/serviceaccount/serviceaccounts_controller.go",
]


def download_code():
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = 0

    for file_path in IMPORTANT_FILES:
        url = f"{RAW_URL}/{file_path}"
        safe_name = file_path.replace("/", "__")
        dest = CODE_DIR / safe_name

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            dest.write_text(response.text)
            downloaded += 1
            print(f"  [{downloaded}] {file_path}")
        except requests.RequestException as e:
            failed += 1
            print(f"  [SKIP] {file_path}: {e}")

    print(f"\nDownloaded {downloaded} Go source files to {CODE_DIR}")
    if failed:
        print(f"  ({failed} files could not be downloaded)")


if __name__ == "__main__":
    download_code()
