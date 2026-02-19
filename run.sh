# Random tensors
./exe/ttid_dim4_cpu Data/random_tensor/Rnd5.tns 12729 100 100 100 100 5000 1e-10 0 0
./exe/ttid_dim5_cpu Data/random_tensor/Rnd6.tns 27843 50 50 50 50 50 15000 1e-10 0 0

./exe/sttid_dim4_cpu Data/random_tensor/Rnd5.tns 12729 100 100 100 100 5000 1e-10 0.1 0 0
./exe/sttid_dim5_cpu Data/random_tensor/Rnd6.tns 27843 50 50 50 50 50 15000 1e-10 0.1 0 0

./exe/sttid_dim4_gpu Data/random_tensor/Rnd5.tns 12729 100 100 100 100 5000 1e-10 0.1 0 0
./exe/sttid_dim5_gpu Data/random_tensor/Rnd6.tns 27843 50 50 50 50 50 15000 1e-10 0.1 0 0

# Knowledge graph tensors
./exe/sttid_dim4_cpu Data/kgraph_tensor/KG_JF17K_3_ids_1based.tns 25820 66 12270 12270 12270 100000 1e-10 0.1 1 1
./exe/sttid_dim5_cpu Data/kgraph_tensor/KG_JF17K_4_ids_1based.tns 15188 50 9528 9528 9528 9528 100000 1e-10 0.1 1 1
./exe/sttid_dim4_cpu Data/kgraph_tensor/KG_WikiPeople_3_ids_1based.tns 34544 104 11541 11541 11541 100000 1e-10 0.1 1 1
./exe/sttid_dim5_cpu Data/kgraph_tensor/KG_WikiPeople_4_ids_1based.tns 9509 23 6536 6536 6536 6536 100000 1e-10 0.1 1 1

./exe/sttid_dim4_gpu Data/kgraph_tensor/KG_JF17K_3_ids_1based.tns 25820 66 12270 12270 12270 100000 1e-10 0.1 1 1
./exe/sttid_dim5_gpu Data/kgraph_tensor/KG_JF17K_4_ids_1based.tns 15188 50 9528 9528 9528 9528 100000 1e-10 0.1 1 1
./exe/sttid_dim4_gpu Data/kgraph_tensor/KG_WikiPeople_3_ids_1based.tns 34544 104 11541 11541 11541 100000 1e-10 0.1 1 1
./exe/sttid_dim5_gpu Data/kgraph_tensor/KG_WikiPeople_4_ids_1based.tns 9509 23 6536 6536 6536 6536 100000 1e-10 0.1 1 1

# FROSTT tensors
./exe/sttid_dim4_cpu Data/frostt_tensor/uber.tns 3309490 183 24 1140 1717 500 1e-10 0.3 0 1
./exe/sttid_dim4_cpu Data/frostt_tensor/nips.tns 3101609 2482 2862 14036 17 500 1e-10 0.3 0 1
./exe/sttid_dim4_cpu Data/frostt_tensor/chicago-crime-comm.tns 5330673 6186 24 77 32 500 1e-10 0.3 0 1
./exe/sttid_dim5_cpu Data/frostt_tensor/chicago-crime-geo.tns 6327013 6185 24 380 395 32 500 1e-10 0.3 0 1

./exe/sttid_dim4_gpu Data/frostt_tensor/uber.tns 3309490 183 24 1140 1717 500 1e-10 0.3 0 1
./exe/sttid_dim4_gpu Data/frostt_tensor/nips.tns 3101609 2482 2862 14036 17 500 1e-10 0.3 0 1
./exe/sttid_dim4_gpu Data/frostt_tensor/chicago-crime-comm.tns 5330673 6186 24 77 32 500 1e-10 0.3 0 1
./exe/sttid_dim5_gpu Data/frostt_tensor/chicago-crime-geo.tns 6327013 6185 24 380 395 32 500 1e-10 0.3 0 1