rm -Rf data/snips_rand_10/
rm data/cached_*
python generate_random_noise.py --task snips --noise 10 --seed 11
python3 main.py --task snips_rand_10 --model_type bert  --model_dir snips_rand_10_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_10/
rm data/cached_*
python generate_random_noise.py --task snips --noise 10 --seed 12
python3 main.py --task snips_rand_10 --model_type bert  --model_dir snips_rand_10_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_10/
rm data/cached_*
python generate_random_noise.py --task snips --noise 10 --seed 13
python3 main.py --task snips_rand_10 --model_type bert  --model_dir snips_rand_10_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6


rm -Rf data/snips_rand_20/
rm data/cached_*
python generate_random_noise.py --task snips --noise 20 --seed 21
python3 main.py --task snips_rand_20 --model_type bert  --model_dir snips_rand_20_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_20/
rm data/cached_*
python generate_random_noise.py --task snips --noise 20 --seed 22
python3 main.py --task snips_rand_20 --model_type bert  --model_dir snips_rand_20_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_20/
rm data/cached_*
python generate_random_noise.py --task snips --noise 20 --seed 23
python3 main.py --task snips_rand_20 --model_type bert  --model_dir snips_rand_20_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6


rm -Rf data/snips_rand_30/
rm data/cached_*
python generate_random_noise.py --task snips --noise 30 --seed 31
python3 main.py --task snips_rand_30 --model_type bert  --model_dir snips_rand_30_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_30/
rm data/cached_*
python generate_random_noise.py --task snips --noise 30 --seed 32
python3 main.py --task snips_rand_30 --model_type bert  --model_dir snips_rand_30_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_30/
rm data/cached_*
python generate_random_noise.py --task snips --noise 30 --seed 33
python3 main.py --task snips_rand_30 --model_type bert  --model_dir snips_rand_30_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_40/
rm data/cached_*
python generate_random_noise.py --task snips --noise 40 --seed 41
python3 main.py --task snips_rand_30 --model_type bert  --model_dir snips_rand_30_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_40/
rm data/cached_*
python generate_random_noise.py --task snips --noise 40 --seed 42
python3 main.py --task snips_rand_40 --model_type bert  --model_dir snips_rand_40_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_40/
rm data/cached_*
python generate_random_noise.py --task snips --noise 40 --seed 43
python3 main.py --task snips_rand_30 --model_type bert  --model_dir snips_rand_30_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_50/
rm data/cached_*
python generate_random_noise.py --task snips --noise 50 --seed 51
python3 main.py --task snips_rand_50 --model_type bert  --model_dir snips_rand_50_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_50/
rm data/cached_*
python generate_random_noise.py --task snips --noise 50 --seed 52
python3 main.py --task snips_rand_50 --model_type bert  --model_dir snips_rand_50_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/snips_rand_50/
rm data/cached_*
python generate_random_noise.py --task snips --noise 50 --seed 53
python3 main.py --task snips_rand_50 --model_type bert  --model_dir snips_rand_50_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6
