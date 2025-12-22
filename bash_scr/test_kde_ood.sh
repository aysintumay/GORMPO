python test_kde_ood_levels.py --model_path /public/gormpo/models/hopper_normal/kde  --dataset_name hopper-medium-v2  --device cuda:0  --distances  0.1 0.3 0.5 0.7 1 --base_path /public/d4rl/ood_test --save_dir figures/kde_ood_distance_tests
 python test_kde_ood_levels.py --model_path /public/gormpo/models/halfcheetah_normal/kde  --dataset_name halfcheetah-medium-v2  --device cuda:0  --distances 0.1 0.3 0.5 0.7 1 --base_path /public/d4rl/ood_test --save_dir figures/kde_ood_distance_tests
 python test_kde_ood_levels.py --model_path /public/gormpo/models/walker2d_normal/kde  --dataset_name walker2d-medium-v2  --device cuda:0  --distances 0.1 0.3 0.5 0.7 --base_path /public/d4rl/ood_test --save_dir figures/kde_ood_distance_tests

python test_realnvp_ood_levels.py --model_path /public/gormpo/models/hopper_normal/realnvp  --dataset_name hopper-medium-v2 
 --device cuda:0  --distances  0.1 0.3 0.5 0.7 1 --base_path /public/d4rl/ood_test --save_dir figures/realnvp_ood_distance_tests
 python test_realnvp_ood_levels.py --model_path /public/gormpo/models/halfcheetah_normal/realnvp  --dataset_name halfcheetah-medium-v2  --device cuda:0  --distances 0.1 0.3 0.5 0.7 1 --base_path /public/d4rl/ood_test --save_dir figures/realnvp_ood_distance_tests
 python test_realnvp_ood_levels.py --model_path /public/gormpo/models/walker2d_normal/realnvp  --dataset_name walker2d-medium-v2  --device cuda:0  --distances 0.1 0.3 0.5 0.7 1 --base_path /public/d4rl/ood_test --save_dir figures/realnvp_ood_distance_tests

python test_vae_ood_levels.py --model_path /public/gormpo/models/hopper_normal/vae  --dataset_name hopper-medium-v2 --device cuda:0  --distances  0.1 0.3 0.5 0.7 1 --base_path /public/d4rl/ood_test --save_dir figures/vae_ood_distance_tests
 python test_vae_ood_levels.py --model_path /public/gormpo/models/halfcheetah_normal/vae  --dataset_name halfcheetah-medium-v2  --device cuda:0  --distances 0.1 0.3 0.5 0.7 1 --base_path /public/d4rl/ood_test --save_dir figures/vae_ood_distance_tests
 python test_vae_ood_levels.py --model_path /public/gormpo/models/walker2d_normal/vae  --dataset_name walker2d-medium-v2  --device cuda:0  --distances 0.1 0.3 0.5 0.7 1 --base_path /public/d4rl/ood_test --save_dir figures/vae_ood_distance_tests
