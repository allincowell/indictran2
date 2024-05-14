Environment Creation

1. conda create -n {env_name} python==3.9
2. conda activate {env_name}
3. bash install.sh

Run 3 scripts

1. bash organize.sh {name of the experiment directory} {src_lang} {tgt_lang}
2. bash prepare_data_joint_training.sh {name of the experiment directory}
3. bash finetune.sh {name of the experiment directory}
