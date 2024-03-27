import json
partition = [17, 21]
stage_to_rank_map = {'0': [0, 1, 2], '1': [3]}
batch = 24
gpu_num = 4
model = 'vgg16'


stage_num = len(stage_to_rank_map)
module_to_stage_map = [i for i in range(0, stage_num)]
module_to_stage_map.append(stage_num-1)
hybrid_conf = {'module_to_stage_map': module_to_stage_map,
               'stage_to_rank_map': stage_to_rank_map}
hybrid_conf_json = json.dumps(hybrid_conf, separators=(',', ':'))

batch_size = [int(batch/len(stage_to_rank_map[chr(i+48)]))
              for i in range(0, stage_num)]
batch_size_all = [batch]
partition_config = {'partition': partition, 'recompute_ratio': [0 for _ in range(
    0, stage_num)], 'batch_size_all': batch_size_all, 'batch_size': batch_size}
partition_config_json = json.dumps(partition_config, separators=(',', ':'))
with open('../runtime/image_classification/models/'+model+'/gpus='+str(gpu_num)+'/hybrid_conf.json', 'w') as hybrid_conf_json_file:
    hybrid_conf_json_file.write(hybrid_conf_json)

with open('../runtime/image_classification/models/'+model+'/gpus='+str(gpu_num)+'/vgg_4.json', 'w') as partition_config_json_file:
    partition_config_json_file.write(partition_config_json)

print(hybrid_conf_json)
print(partition_config_json)
