CUDA=5
property=homo

CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_conditional_qm9.py version=${property}_cfg1.0_1 classifiers_path='../checkpoints/classifiers_ckpts/exp_class_${property}' property=${property} iterations=9 batch_size=1000 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_qm9_frad_${property}/model/checkpoint-last.pth 

CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_conditional_qm9.py version=${property}_cfg1.0_2 classifiers_path='../checkpoints/classifiers_ckpts/exp_class_${property}' property=${property} iterations=9 batch_size=1000 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_qm9_frad_${property}/model/checkpoint-last.pth 

CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_conditional_qm9.py version=${property}_cfg1.0_3 classifiers_path='../checkpoints/classifiers_ckpts/exp_class_${property}' property=${property} iterations=9 batch_size=1000 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_qm9_frad_${property}/model/checkpoint-last.pth  