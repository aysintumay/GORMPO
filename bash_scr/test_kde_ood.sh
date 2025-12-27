# python test_kde_ood_levels.py --model_path /public/gormpo/models/hopper_normal/kde  --dataset_name hopper-medium-v2  --device cuda:0  --distances  1 5 10 15 --base_path /public/d4rl/ood_test --save_dir figures/kde_ood_distance_tests
#  python test_kde_ood_levels.py --model_path /public/gormpo/models/halfcheetah_normal/kde  --dataset_name halfcheetah-medium-v2  --device cuda:0  --distances 1 5 10 15 --base_path /public/d4rl/ood_test --save_dir figures/kde_ood_distance_tests
#  python test_kde_ood_levels.py --model_path /public/gormpo/models/walker2d_normal/kde  --dataset_name walker2d-medium-v2  --device cuda:0  --distances 0.1 0.3 0.5 0.7 --base_path /public/d4rl/ood_test --save_dir figures/kde_ood_distance_tests

# python test_realnvp_ood_levels.py --model_path /public/gormpo/models/hopper_normal/realnvp  --dataset_name hopper-medium-v2 
#  --device cuda:0  --distances  1 5 10 15 --base_path /public/d4rl/ood_test --save_dir figures/realnvp_ood_distance_tests
#  python test_realnvp_ood_levels.py --model_path /public/gormpo/models/halfcheetah_normal/realnvp  --dataset_name halfcheetah-medium-v2  --device cuda:0  --distances 1 5 10 15 --base_path /public/d4rl/ood_test --save_dir figures/realnvp_ood_distance_tests
#  python test_realnvp_ood_levels.py --model_path /public/gormpo/models/walker2d_normal/realnvp  --dataset_name walker2d-medium-v2  --device cuda:0  --distances 1 5 10 15 --base_path /public/d4rl/ood_test --save_dir figures/realnvp_ood_distance_tests

# python test_vae_ood_levels.py --model_path /public/gormpo/models/hopper_normal/vae  --dataset_name hopper-medium-v2 --device cuda:0  --distances  1 5 10 15 --base_path /public/d4rl/ood_test --save_dir figures/vae_ood_distance_tests
#  python test_vae_ood_levels.py --model_path /public/gormpo/models/halfcheetah_normal/vae  --dataset_name halfcheetah-medium-v2  --device cuda:0  --distances 1 5 10 15 --base_path /public/d4rl/ood_test --save_dir figures/vae_ood_distance_tests
#  python test_vae_ood_levels.py --model_path /public/gormpo/models/walker2d_normal/vae  --dataset_name walker2d-medium-v2  --device cuda:0  --distances 1 5 10 15 --base_path /public/d4rl/ood_test --save_dir figures/vae_ood_distance_tests

# python test_neuralode_ood_levels.py --model_path neuralODE/hopper_ood --dataset_name hopper-medium-v2 --target_dim 14 --device cuda:0 --distances 1 5 10 15 --save_dir figures/neuralODE_OOD
# python test_neuralode_ood_levels.py --model_path neuralODE/halfcheetah_ood --dataset_name halfcheetah-medium-v2 --target_dim 23 --device cuda:0 --distances 1 5 10 15 --save_dir figures/neuralODE_OOD
# python test_neuralode_ood_levels.py --model_path neuralODE/walker2d_ood --dataset_name walker2d-medium-v2 --target_dim 23 --device cuda:0 --distances 1 5 10 15 --save_dir figures/neuralODE_OOD

conda deactivate
source venv/bin/activate

python test_diffusion_ood_levels.py --model_path /public/gormpo/models/hopper/diffusion/checkpoint.pt --scheduler_dir /public/gormpo/models/hopper/diffusion/scheduler --dataset_name hopper-medium-v2 --device cuda:0 --distances 1 5 10 15 --num_inference_steps 20 --save_dir figures/diffusion_ood_distance_tests
python test_diffusion_ood_levels.py --model_path /public/gormpo/models/halfcheetah/diffusion/checkpoint.pt --scheduler_dir /public/gormpo/models/halfcheetah/diffusion/scheduler --dataset_name halfcheetah-medium-v2 --device cuda:0 --distances 1 5 10 15 --num_inference_steps 20 --save_dir figures/diffusion_ood_distance_tests
python test_diffusion_ood_levels.py --model_path /public/gormpo/models/walker2d/diffusion/checkpoint.pt --scheduler_dir /public/gormpo/models/walker2d/diffusion/scheduler --dataset_name walker2d-medium-v2 --device cuda:0 --distances 1 5 10 15 --num_inference_steps 20 --save_dir figures/diffusion_ood_distance_tests
