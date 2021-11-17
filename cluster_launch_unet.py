import os

from glob import glob

ROOT_DIR = '/groups/stringer/stringerlab/PoseEstimation/moco_unet/'
CHECKPOINT_DIR = 'checkpoints'
DATA_DIR = 'data/supervised/longw'
AUG_FILE = 'aug_params.yml'

LR = 0.001
EPOCHS = 400

MOCO_PREFIX = 'moco_unet'
SAVE_PREFIX = f'unet_pretrained_longw_{LR}_{EPOCHS}'
CHECKPOINT_NAME = 'checkpoint_0200.pth.tar'


def main():
    os.chdir(ROOT_DIR)

    base_bsub_command = f'bsub -n 4 -gpu "num=1" -q gpu_rtx -W 240'
    base_python_command = f'python code/main_unet.py --batch-size 8 --lr {LR} --epochs {EPOCHS}'
    
    model_paths = glob(os.path.join(CHECKPOINT_DIR, f'{MOCO_PREFIX}*'))
    save_dirs = []
    for model_path in model_paths:
        checkpoint = os.path.join(model_path, CHECKPOINT_NAME)
        m_ids = model_path.split('_')
        start = m_ids.index('0.003')
        end = m_ids.index('cos')
        save_id = m_ids[start+1:end]
        save_name =  f'{SAVE_PREFIX}_{"_".join(save_id)}'
        save_dir = os.path.join(CHECKPOINT_DIR, save_name)
        output_file = os.path.join(save_dir, 'output.txt')
        error_file = os.path.join(save_dir, 'error.txt')
        
        python_command = f'{base_python_command} --pretrained {checkpoint} {DATA_DIR} {save_dir} {AUG_FILE}'
        bsub_command = f'{base_bsub_command} -J {save_name} -o {output_file} -e {error_file} "{python_command}"'
       
        os.makedirs(save_dir, exist_ok=True)
        os.system(bsub_command)
        print(bsub_command, end='\n\n')


if __name__ == '__main__':
    main()
    
