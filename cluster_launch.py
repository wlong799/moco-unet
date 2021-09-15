import os
import socket

ROOT_DIR = '/groups/stringer/stringerlab/PoseEstimation/moco-unet/'
# ROOT_DIR = '/home/labadmin/Documents/moco-unet'
CHECKPOINTS_DIR = 'checkpoints'
CODE_DIR = 'code'
DATA_DIR = 'data'

NUM_GPUS = 4
WALLTIME = 600

START_PORT = 10001
CONNECT_ATTEMPTS = 100


def main():
    os.chdir(ROOT_DIR)
    opt_args = {
        'learning-rate': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        # 'moco-k': [128, 256, 512, 1024],
        # 'mlp': [False, True],
        # 'cos': [False, True]
    }
    datasets = ['unsupervised_8k']

    base_bsub_command = f'bsub -n {NUM_GPUS} -gpu "num={NUM_GPUS}" -q gpu_rtx -W {WALLTIME}'
    base_python_command = f'python code/main_moco.py -j {NUM_GPUS * 2} -b {NUM_GPUS * 16}'

    # write optional arguments
    python_commands = [base_python_command]
    save_names = ['moco_unet']
    for i, arg_name in enumerate(opt_args):
        arg_vals = opt_args[arg_name]
        new_python_commands = []
        new_save_names = []
        for arg_val in arg_vals:
            for (command, save_name) in zip(python_commands, save_names):
                if type(arg_val) in (int, float):
                    new_python_commands.append(command + f' --{arg_name} {arg_val}')
                    new_save_names.append(save_name + f'_{arg_val}')
                elif type(arg_val) is bool and arg_val:
                    new_python_commands.append(command + f' --{arg_name}')
                    new_save_names.append(save_name + f'_{arg_name}')
                elif type(arg_val) is bool:
                    new_python_commands.append(command)
                    new_save_names.append(save_name)
        python_commands = new_python_commands
        save_names = new_save_names

    # write port, data directory, and checkpoint directory and remaining bsub args
    port = START_PORT
    bsub_commands = []
    for dataset in datasets:
        data_dir = os.path.join(DATA_DIR, dataset)
        for command, save_name in zip(python_commands, save_names):
            attempts = 0
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                while s.connect_ex(('localhost', port)) == 0 and attempts < CONNECT_ATTEMPTS:
                    port += 1
                    attempts += 1
            save_dir = os.path.join(CHECKPOINTS_DIR, save_name + f'_{dataset}')
            os.makedirs(save_dir, exist_ok=True)
            output_file = os.path.join(save_dir, 'output.txt')
            error_file = os.path.join(save_dir, 'error.txt')
            python_command = command + f' --port {port} {data_dir} {save_dir}'
            command = base_bsub_command + f' -J moco_unet_{port} -o {output_file} -e {error_file} "{python_command}"'
            bsub_commands.append(command)
            port += 1

    for command in bsub_commands:
        os.system(command)
        print(command)


if __name__ == '__main__':
    main()
