rm -Rf data/atis_rand_10/
rm data/cached_*
python generate_random_noise.py --task atis --noise 10 --seed 1
python3 main.py --task atis_rand_10 --model_type bert  --model_dir atis_rand_10_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/atis_rand_10/
rm data/cached_*
python generate_random_noise.py --task atis --noise 10 --seed 2
python3 main.py --task atis_rand_10 --model_type bert  --model_dir atis_rand_10_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/atis_rand_10/
rm data/cached_*
python generate_random_noise.py --task atis --noise 10 --seed 3
python3 main.py --task atis_rand_10 --model_type bert  --model_dir atis_rand_10_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6


rm -Rf data/atis_rand_20/
rm data/cached_*
python generate_random_noise.py --task atis --noise 20 --seed 1
python3 main.py --task atis_rand_20 --model_type bert  --model_dir atis_rand_20_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/atis_rand_20/
rm data/cached_*
python generate_random_noise.py --task atis --noise 20 --seed 2
python3 main.py --task atis_rand_20 --model_type bert  --model_dir atis_rand_20_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/atis_rand_20/
rm data/cached_*
python generate_random_noise.py --task atis --noise 20 --seed 3
python3 main.py --task atis_rand_20 --model_type bert  --model_dir atis_rand_20_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6


rm -Rf data/atis_rand_30/
rm data/cached_*
python generate_random_noise.py --task atis --noise 30 --seed 1
python3 main.py --task atis_rand_30 --model_type bert  --model_dir atis_rand_30_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/atis_rand_30/
rm data/cached_*
python generate_random_noise.py --task atis --noise 30 --seed 2
python3 main.py --task atis_rand_30 --model_type bert  --model_dir atis_rand_30_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6

rm -Rf data/atis_rand_30/
rm data/cached_*
python generate_random_noise.py --task atis --noise 30 --seed 3
python3 main.py --task atis_rand_30 --model_type bert  --model_dir atis_rand_30_model  --do_train --do_eval --num_train_epochs 25 --learning_rate 5e-6
