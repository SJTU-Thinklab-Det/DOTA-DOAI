import os

res_dir = '/data/yangxue/code/DOTA_DOAI/FPN_Tensorflow_Rotation/tools/test_dota/FPN_Res152D_DOTA1.0_20191106_v1/dota_res'
pseudo_label_dir = '/data/yangxue/dataset/DOTA/test/labelTxt'

res_list = os.listdir(res_dir)

for t in res_list:
    with open(os.path.join(res_dir, t), 'r') as fr:
        lines = fr.readlines()
        for l in lines:
            res = l.split(' ')
            if float(res[1]) > 0.4:
                if not os.path.exists(os.path.join(pseudo_label_dir, res[0] + '.txt')):
                    fw = open(os.path.join(pseudo_label_dir, res[0] + '.txt'), 'w')
                else:
                    fw = open(os.path.join(pseudo_label_dir, res[0] + '.txt'), 'a+')
                command = '{} {} {} {} {} {} {} {} {}\n'.format(res[2], res[3], res[4], res[5],
                                                                res[6], res[7], res[8], res[9].split('\n')[0], t.split('_')[-1].split('.')[0])
                fw.write(command)
                fw.close()

