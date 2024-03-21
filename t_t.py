import copy
import torch
import numpy
# def runtime_control(layers,stages,num_layer,present_stage_id,start_id):
#     #layers(id,stage_id,forward_time,backward_time,communication)
#     #stages(id,compute_time)
#     #num_layer(id,num)
#     #start_id 当前stage起始层的id
#     record=[[0 for _ in range(num_layer[present_stage_id]+1)] for _ in range(num_layer[present_stage_id]+1)]
#     def find_min_index_2d(arr):
#         min_value = float('inf')
#         min_index = (0, 0)
#         for i in range(len(arr)):
#             for j in range(len(arr[i])):
#                 if arr[i][j] < min_value:
#                     min_value = arr[i][j]
#                     min_index = (i, j)
#         return min_index
#     def compute_time_1(begin_layer_id,end_layer_id):
#         stages_=copy.copy(stages)
#         list_ = []
#         for i in range(end_layer_id - begin_layer_id):
#             stages_[present_stage_id - 1] += layers[start_id + i]
#         for i in range(0, present_stage_id):
#             list_.append(stages_[i])
#         max_time = max(list_)
#         return max_time
#
#     def compute_time_2(begin_layer_id, end_layer_id):
#         stages_=copy.copy(stages)
#         list_ = []
#         for i in range(end_layer_id - begin_layer_id):
#             stages_[present_stage_id + 1] += layers[start_id + begin_layer_id + i]
#         for i in range(present_stage_id + 1, 4):
#             # change 4 to stage_num_sum
#             list_.append(stages_[i])
#         max_time = max(list_)
#         return max_time
#
#     def compute_time_3(begin_layer_id, end_layer_id):
#         stages_ = copy.copy(stages)
#         stages_[present_stage_id] = 0
#         for i in range(end_layer_id - begin_layer_id):
#             stages_[present_stage_id] += layers[start_id + begin_layer_id + i]
#         return stages_[present_stage_id]
#     for i in range(num_layer[present_stage_id]+1):
#         for j in range(num_layer[present_stage_id]+1):
#             if i>=j:
#                 record[i][j]=float('inf')
#             else:
#
#                 time1=compute_time_1(0,i)#pre part
#                 time2=compute_time_3(i,j)#medium part f+b
#                 time3=compute_time_2(j,num_layer[present_stage_id])#last part
#                 record[i][j]=max(time1,time2,time3)
#     min_index=find_min_index_2d(record)
#     return min_index


def runtime_control(layers,stages,num_layer,present_stage_id,start_id,communicaiton,straggle):
    #layers(id,stage_id,forward_time,backward_time,communication) list
    #stages(id,compute_time) list
    #num_layer(id,num) list
    #start_id 所有stage的起始层id list
    #communication list
    record=[[0 for _ in range(num_layer[present_stage_id]+1)] for _ in range(num_layer[present_stage_id]+1)]
    def compute_pipline_time(stage_list,communication_list):
        if len(stage_list)==1:
            return stage_list[0]
        for i in range(len(stage_list)-1):
            tmp1=stage_list[i]
            tmp2=stage_list[i+1]
            tmp_communication=communication_list[i]
            tmp_time=0
            if tmp1==max(tmp1,tmp2):
                if (tmp1)/2>=(tmp2)/2+tmp_communication:
                    tmp_time=tmp1
                else:
                    tmp_time=tmp_communication-(tmp1-tmp2)/2+tmp1
            if tmp2==max(tmp1,tmp2):
                if (tmp2)/2>=(tmp1)/2+tmp_communication:
                    tmp_time=tmp2
                else:
                    tmp_time=tmp_communication-(tmp2-tmp1)/2+tmp2
            stage_list[i+1]=tmp_time
        return stage_list[-1]
    def find_min_index_2d(arr):
        min_value = float('inf')
        min_index = (0, 0)
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                if arr[i][j] < min_value:
                    min_value = arr[i][j]
                    min_index = (i, j)
        return min_index
    def compute_time_1(begin_layer_id,end_layer_id):
        stages_=copy.copy(stages)
        list_ = []
        communication_list=[]
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id - 1] += layers[start_id[present_stage_id] + i]*straggle[present_stage_id-1]
        for i in range(0, present_stage_id):
            list_.append(stages_[i])
            communication_list.append(communicaiton[i])
        max_time = compute_pipline_time(list_,communication_list)
        return max_time

    def compute_time_2(begin_layer_id, end_layer_id):
        stages_=copy.copy(stages)
        list_ = []
        communication_list = []
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id + 1] += layers[start_id[present_stage_id] + begin_layer_id + i]*straggle[present_stage_id+1]
        for i in range(present_stage_id + 1, 4):
            # change 4 to stage_num_sum
            list_.append(stages_[i])
            communication_list.append(communicaiton[i])
        max_time = compute_pipline_time(list_, communication_list)
        return max_time

    def compute_time_3(begin_layer_id, end_layer_id):
        stages_ = copy.copy(stages)
        stages_[present_stage_id] = 0
        for i in range(end_layer_id - begin_layer_id):
            stages_[present_stage_id] += layers[start_id[present_stage_id] + begin_layer_id + i]*straggle[present_stage_id]
        return stages_[present_stage_id]
    for i in range(num_layer[present_stage_id]+1):
        for j in range(num_layer[present_stage_id]+1):
            if i>=j:
                record[i][j]=float('inf')
            else:
                if present_stage_id-1>=0:
                    time1=compute_time_1(0,i)#pre part
                else:
                    if i!=0:
                        record[i][j] = float('inf')
                        continue
                    time1=0
                time2=compute_time_3(i,j)#medium part f+b
                if present_stage_id+1<=3:
                    #change 3 to last stage id
                    time3=compute_time_2(j,num_layer[present_stage_id])#last part
                else:
                    if j!=num_layer[present_stage_id]:
                        record[i][j]=float('inf')
                        continue
                    time3=0
                if time1==0:
                    record[i][j]=compute_pipline_time([time2,time3],[communicaiton[j-1]])
                elif time3==0:
                    record[i][j]=compute_pipline_time([time1,time2],[communicaiton[start_id[present_stage_id]+i-1]])
                else:
                    tmp=compute_pipline_time([time1,time2],[communicaiton[start_id[present_stage_id]+i-1]])
                    record[i][j]=compute_pipline_time([tmp,time3],[communicaiton[start_id[present_stage_id]+j-1]])
    min_index=find_min_index_2d(record)
    print(record[min_index[0]][min_index[1]])
    return min_index

if __name__ == '__main__':
    #layer=[2,3,4,1,5,6,2,1,2,1]
    layer=[1,1,1,1,1,1,1,1,1,1]
    stage=[]
    stages=torch.tensor([1,1,3,5])
    start_id=[0,1,2,5]
    #communication=[1,2,1,2,1,2,1,2,1]
    communication=[0,0,0,0,0,0,0,0,0]
    stage=stages.numpy()
    #stage=[5,10,6,6]
    num_layer=[1,1,3,5]
    straggle=[1,1,1,1]
    #tmp=runtime_control(layer,stage,num_layer,1,3)
    # tmp=runtime_control(layer,stage,num_layer,3,start_id,communication,straggle)
    # print(tmp)


    while True:
        max_index=numpy.argmax(stage)
        tmp=runtime_control(layer,stage,num_layer,max_index,start_id,communication,straggle)
        if tmp[0]==0 and tmp[1]==num_layer[max_index]:
            print(num_layer)
            break
        if tmp[0]!=0:
            for i in range(tmp[0]):
                stage[max_index-1]+=layer[start_id[max_index]+i]*straggle[max_index-1]
            num_layer[max_index-1]+=tmp[0]
        if tmp[1]!=num_layer[max_index]:
            for i in range(tmp[1],num_layer[max_index]):
                stage[max_index+1]+=layer[start_id[max_index]+i]*straggle[max_index+1]
            num_layer[max_index+1]+=num_layer[max_index]-tmp[1]
            start_id[max_index+1]=start_id[max_index+1]-(num_layer[max_index]-tmp[1])
        stage[max_index]=0
        for i in range(start_id[max_index]+tmp[0],start_id[max_index]+tmp[1]):
            stage[max_index]+=layer[i]*straggle[max_index]
        num_layer[max_index]=tmp[1]-tmp[0]
        start_id[max_index]=start_id[max_index]+tmp[0]
