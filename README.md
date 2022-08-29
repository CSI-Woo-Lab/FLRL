# 0. Pre-setting

0.1 Download FLRL code : [git@github.com](mailto:git@github.com):jhok0623/FLRL.git

0.2 Install the necessities :

- gym,
- navigation_2d,
- stable_baseline3
- tensorboard
- Before anything, **change goal position**
    - navigation_2d.navigation_env.drone_start.pos
        
        Currently: [0.5, 0.5]
        
        **Change to: [0.5. 0.924]**
        

# 1. Train expert

### 1.1 train_expert.py

example) python train_expert.py —env_id 0

: train two differerent goals (0 & 3 for instance)

→ this would create ‘tensorboard’, ‘model_env_id_n’ in the directory

_______________________________________________

# 2. Prepare offline dataset

### 2.0 install

- tqdm
- d3rlpy

### 2.1 fl_gather_buffer.py

** CHANGE SEED BEFOREHAND

example) python fl_gather_buffers.py —env_id 0 —expert_steps 100000 —num_trajectories 100 — num_clients 2

: gather {num_clients} sets of offline datasets of environment {env_id} 

### 2.2 fl_buffer_to_mdp

** CHANGE SEED BEFOREHAND

example) python fl_buffer_to_mdp.py —env_id 0 —num_trajectories 100 —num_clients 2

: transform the datasets into mdp form

_________________________________________________

# 3. Offline FRL

### 3.0 install & add LR scheduler

- flwr
- torch
- d3rlpy/algos/cql.py
    - import torch
    - after self._impl.build()
    - create scheduler
        
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._impl._actor_optim, eta_min=0.0001, T_max=5)
        
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._impl._critic_optim, eta_min=0.0001, T_max=5)
        
        self.temp_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._impl._temp_optim, eta_min=0.0001, T_max=5)
        
        self.alpha_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._impl._alpha_optim, eta_min=0.0001, T_max=5)
        
        self.actor_scheduler = torch.optim_lr_scheduler.CyclicLR(self._impl._actor_optim, base_lr=0.0001, max_lr=0.0025, step_size_up=5, cycle_momentum=False)
        
        self.critic_scheduler = torch.optim.lr_scheduler.CyclicLR(self._impl._critic_optim, base_lr=0.0001, max_lr=0.0025, step_size_up=5, cycle_momentum=False)
        
        self.temp_scheduler = torch.optim_lr_scheduler.CyclicLR(self._impl._temp_optim, base_lr=0.0001, max_lr=0.0025, step_size_up=5, cycle_momentum=False)
        
        self.alpha_scheduler = torch.optim.lr_scheduler.CyclicLR(self._impl._alpha_optim, base_lr=0.0001, max_lr=0.0025, step_size_up=5, cycle_momentum=False)
        
    - before metrics
    - create step
        
        self.actor_scheduler.step()
        
        self.critic_scheduler.step()
        
        self.temp_scheduler.step()
        
        self.alpha_scheduler.step()
        
- d3rlpy/logger.py
    - import torch
    - add type in ‘default_json_encoder’
        
        elif isinstance(obj, torch.optim.lr_scheduler.CosineAnnealingLR):
        
        return ‘scheduler’
        
        elif isinstance(obj, torch.optim.lr_scheduler.CyclicLR):
        
        return ‘scheduler’
        

### 3.1 flserver.py

example) python fl_server.py —env_id 0

: initiate a server

### 3.2. flclient.py

** change learning_rate_scheduler

example) python fl_client.py —num_trajectories 100 —seed 100 —env_id 0

# 4. Evaluation

### 4.0 install

- cv2 (pip install openv-python)

### 4.1 eval.py

- change ‘main_dir_name’

example) python [eval.py](http://eval.py) —env_id 0

: eval on both environment (0&3 for instance)

- this would create ‘task{task_id}.csv’ file
# FLRL
# FLRL
