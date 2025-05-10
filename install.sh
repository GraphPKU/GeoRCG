conda create -n georcg python=3.8 -y
conda activate georcg

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install -r requirements.yml


wget https://github.com/dptech-corp/Uni-Core/releases/download/0.0.3/unicore-0.0.1+cu118torch2.0.0-cp38-cp38-linux_x86_64.whl
pip install unicore-0.0.1+cu118torch2.0.0-cp38-cp38-linux_x86_64.whl
rm unicore-0.0.1+cu118torch2.0.0-cp38-cp38-linux_x86_64.whl




