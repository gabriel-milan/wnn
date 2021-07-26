from random import sample
from os import listdir, path
from subprocess import run
from os.path import isfile, join

SUBSET_SIZE = 100
CONFIGS_PATH = "ensemble_configs/"


def print_and_run(cmd: str):
    print(f"Executing command\"{cmd}\"...")
    run(cmd.split(" "))


config_files = [path.abspath(CONFIGS_PATH + f) for f in listdir(
    CONFIGS_PATH) if isfile(join(CONFIGS_PATH, f))]
command = "/usr/bin/sbatch --job-name={job_name}.job -n 1 --mem-per-cpu=8G -c 1 /home/gabriel-milan/GIT_REPOS/wnn/scripts/run_ensemble.sh {config_file}"

i = 0

if SUBSET_SIZE == -1:
    for config_file in config_files:
        cmd = command.format(job_name="ensemble_{}_{}".format(i, config_file.split("/")[-1]),
                             config_file=config_file)
        print_and_run(cmd)
        i += 1
else:
    for config_file in sample(config_files, k=SUBSET_SIZE):
        cmd = command.format(job_name="ensemble_{}_{}".format(i, config_file.split("/")[-1]),
                             config_file=config_file)
        print_and_run(cmd)
        i += 1
