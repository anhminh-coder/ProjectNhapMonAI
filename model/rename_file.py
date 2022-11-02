import os

data_path = os.getcwd()+'/dataset'
for name in os.listdir(data_path):
    img_folder_path = data_path+'/'+name
    id_ = 1
    for file in os.listdir(img_folder_path):
        old_name = img_folder_path+'/'+file
        new_name = img_folder_path+'/'+str(id_)+'.jpg'
        os.rename(old_name, new_name)
        id_ += 1
