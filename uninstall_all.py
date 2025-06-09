import pkg_resources
import subprocess

packages = [pkg.key for pkg in pkg_resources.working_set]
for pkg in packages:
    subprocess.call(["pip", "uninstall", "-y", pkg])
