# Oxford-IIIT Pet Dataset https://www.robots.ox.ac.uk/~vgg/data/pets/
# Download command: bash get_oxford_pet.sh

# Download, unzip, remove
mkdir dataset && cd dataset
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
tar -zxvf images.tar.gz && rm images.tar.gz &
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz
tar -zxvf annotations.tar.gz && rm annotations.tar.gz &