### 自监督学习和监督学习的对比

* 在./data目录执行：get_data.sh 获得数据
* 执行：bash run_train.sh 训练模型和评估
* ./ckp保存模型参数，./args记录命令输入的超参数，./logs保存tensorboard数据
* mytrain.py文件是训练文件，也集成了进行Linear classification protocol的部分（evaluate.py）