def calculate_new_placement(layer_forward_list,layer_backward_list,layer_communication_list,straggle_for_stage,stage_num,stage_nums,top_k):
    def main(stage_num,forward_cost_list,backward_cost_list,comm_cost_list,max_micro_batch_num,cur_micro_batch_num,timestamp):

        # # Total stage number
        # stage_num = 2
        # # Forward cost and backward cost list of each stage
        # forward_cost_list = [1, 1]
        # backward_cost_list = [3, 3]
        # # Communication cost list of each stage. e.g. The first element is the communication cost between stage 1 and stage 2.
        # comm_cost_list = [1]
        #
        # max_micro_batch_num = 99
        # cur_micro_batch_num = 0
        # timestamp = 0

        def get_comm_cost(stage1: float, stage2: float) -> float:
            """计算两个相邻 stage 的通信时间 (stage 顺序不做限制)。

            Args:
                stage1 (int): 前一个 stage 的编号。
                stage2 (int): 后一个 stage 的编号。

            Returns:
                int: _description_
            """
            assert abs(stage1 - stage2) == 1, "The stages must be adjacent!"
            return comm_cost_list[min(stage1, stage2)]

        class Stage:
            def __init__(self, idx: int, forward_cost: float, backward_cost: float) -> None:
                self.idx = idx
                self.forward_cost = forward_cost
                self.backward_cost = backward_cost

                self.warmup_mb_num = -1  # not used yet

                self.f_idx = -1  # index of next forward micro_batch
                self.b_idx = -1  # index of next backward micro_batch

            def __str__(self) -> str:
                return f"stage_idx: {self.idx}\tforward_cost: {self.forward_cost}\tbackward_cost: {self.backward_cost}\tf_idx: {self.f_idx}\tb_idx: {self.b_idx}."

            def get_next_f_idx(self) -> int:
                assert self.b_idx >= 0, "Previous backward index is not initialized!"
                return self.b_idx + (stage_num - self.idx)

            def get_next_b_idx(self) -> int:
                assert self.f_idx >= 0, "Previous forward index is not initialized!"
                return self.f_idx - (stage_num - 1 - self.idx)

            def get_prev_b_idx(self) -> int:
                assert self.f_idx >= 0, "Current forward index is not initialized!"
                return self.f_idx - (stage_num - self.idx)

            def get_prev_f_idx(self) -> int:
                assert self.b_idx >= 0, "Current backward index is not initialized!"
                return self.b_idx + (stage_num - 1 - self.idx)

        def generate_stages():
            stages = [Stage(i, forward_cost_list[i], backward_cost_list[i]) for i in range(stage_num)]
            return stages

        stages = generate_stages()

        class MicroBatch:
            def __init__(self, stage_num: int) -> None:
                self.stage_num = stage_num
                self.forward_ts = -1
                self.backward_ts = -1

                self.forward_cost = stages[self.stage_num].forward_cost
                self.backward_cost = stages[self.stage_num].backward_cost

            def __str__(self) -> str:
                return f"stage: {self.stage_num}\tforward_ts: {self.forward_ts}\tbackward_ts: {self.backward_ts}."

        def generate_batches():
            batches = [
                [
                    MicroBatch(stage)
                    for _ in range(max_micro_batch_num)
                ] \
                for stage in range(stage_num)
            ]
            return batches

        def get_warmup_micro_batch_num(stage: int) -> int:
            return stage_num - stage - 1

        batches = generate_batches()

        def show_batches(show_diff: bool = True):
            last_forward_ts, last_backward_ts = -1, -1
            for stage_idx in range(stage_num):
                print(f"===== stage {stage_idx} =====")
                print(type(batches[stage_idx]),len(batches[stage_idx]))
                for idx, stage in enumerate(batches[stage_idx]):
                    if idx==len(batches[stage_idx])-stage_num:
                        return
                    if show_diff:
                        if idx % stage_num == 0:
                            if idx != 0:

                                print(
                                    f"mb_index: {idx}\tforward_diff: {stage.forward_ts - last_forward_ts}\tbackward_diff: {stage.backward_ts - last_backward_ts}"
                                )
                            last_forward_ts = stage.forward_ts
                            last_backward_ts = stage.backward_ts
                    else:
                        print(f"mb_index: {idx}", stage)

        def show_batches_time(show_diff: bool = True):
            last_forward_ts, last_backward_ts = -1, -1
            time_sum=0
            count=0
            for stage_idx in range(1):
                for idx, stage in enumerate(batches[stage_idx]):
                    if idx==len(batches[stage_idx])-stage_num:
                        return time_sum/(stage_num*count)
                    if show_diff:
                        if idx % stage_num == 0:
                            if idx != 0:
                                count+=1
                                time_sum+=stage.forward_ts - last_forward_ts
                            last_forward_ts = stage.forward_ts
                    else:
                        print(f"mb_index: {idx}", stage)

        def show_stages():
            for stage in stages:
                print(stage)

        # * 1. Warmup stage
        for stage in range(stage_num):
            warmup_micro_batch_num = get_warmup_micro_batch_num(stage)
            stages[stage].warmup_mb_num = warmup_micro_batch_num
            if stage != stage_num - 1:
                if stage == 0:
                    batches[stage][0].forward_ts = timestamp

                for i in range(warmup_micro_batch_num):
                    if stage == 0:
                        if i != 0:
                            batches[stage][i].forward_ts = \
                                batches[stage][i - 1].forward_ts + batches[stage][i - 1].forward_cost
                    else:
                        batches[stage][i].forward_ts = max(
                            batches[stage][i - 1].forward_ts + batches[stage][i - 1].forward_cost,
                            batches[stage - 1][i].forward_ts + batches[stage - 1][i].forward_cost +
                            get_comm_cost(stage - 1, stage)
                        )
                # update f_idx
                stages[stage].f_idx = warmup_micro_batch_num
            else:
                # update f_idx
                stages[stage].f_idx = 0

        # show_batches()
        # show_stages()

        # * 2. Running 1F1B
        is_f_mode = True
        exit_flag = False
        while cur_micro_batch_num <= max_micro_batch_num and (not exit_flag):
            # * Forward pass
            if is_f_mode:
                for stage_idx in range(stage_num):
                    stage = stages[stage_idx]

                    prev_b_idx = stage.get_prev_b_idx()
                    prev_backward_finish_ts = -1 if prev_b_idx < 0 else \
                        batches[stage_idx][prev_b_idx].backward_ts + batches[stage_idx][prev_b_idx].backward_cost

                    if stage_idx == 0:
                        batches[stage_idx][stage.f_idx].forward_ts = max(
                            batches[stage_idx][stage.f_idx - 1].forward_ts +
                            batches[stage_idx][stage.f_idx - 1].forward_cost, prev_backward_finish_ts
                        )
                    elif stage_idx == stage_num - 1:
                        batches[stage_idx][stage.f_idx].forward_ts = max(
                            batches[stage_idx - 1][stage.f_idx].forward_ts +
                            batches[stage_idx - 1][stage.f_idx].forward_cost + get_comm_cost(stage_idx - 1, stage_idx),
                            prev_backward_finish_ts
                        )
                    else:
                        batches[stage_idx][stage.f_idx].forward_ts = max(
                            prev_backward_finish_ts,
                            max(
                                batches[stage_idx][stage.f_idx - 1].forward_ts +
                                batches[stage_idx][stage.f_idx - 1].forward_cost,
                                batches[stage_idx - 1][stage.f_idx].forward_ts +
                                batches[stage_idx - 1][stage.f_idx].forward_cost + get_comm_cost(stage_idx - 1, stage_idx)
                            )
                        )
                    # update cur_micro_batch_num
                    cur_micro_batch_num = max(cur_micro_batch_num, stage.f_idx)
                    # update b_idx
                    next_b_idx = stage.get_next_b_idx()
                    # print(f"next_b_idx: {next_b_idx}")
                    stage.b_idx = next_b_idx
                    if stage.b_idx >= max_micro_batch_num:
                        exit_flag = True
                        break

            # * Backward pass
            else:
                for stage_idx in range(stage_num - 1, -1, -1):
                    stage = stages[stage_idx]

                    prev_f_idx = stage.get_prev_f_idx()
                    prev_forward_finish_ts = -1 if prev_f_idx < 0 else \
                        batches[stage_idx][prev_f_idx].forward_ts + batches[stage_idx][prev_f_idx].forward_cost

                    if stage_idx == 0:
                        batches[stage_idx][stage.b_idx].backward_ts = max(
                            prev_forward_finish_ts,\
                            batches[stage_idx + 1][stage.b_idx].backward_ts + batches[stage_idx + 1][stage.b_idx].backward_cost + get_comm_cost(stage_idx, stage_idx + 1)
                        )
                    elif stage_idx == stage_num - 1:
                        batches[stage_idx][stage.b_idx].backward_ts = prev_forward_finish_ts
                    else:
                        batches[stage_idx][stage.b_idx].backward_ts = max(
                            prev_forward_finish_ts,\
                            batches[stage_idx + 1][stage.b_idx].backward_ts + batches[stage_idx + 1][stage.b_idx].backward_cost + get_comm_cost(stage_idx, stage_idx + 1)
                        )
                    # update cur_micro_batch_num
                    cur_micro_batch_num = max(cur_micro_batch_num, stage.b_idx)
                    # update f_idx
                    next_f_idx = stage.get_next_f_idx()
                    # print(f"prev_f_idx: {stage.f_idx}, next_f_idx: {next_f_idx}")
                    stage.f_idx = next_f_idx
                    if stage.f_idx >= max_micro_batch_num:
                        exit_flag = True
                        break

            # change mode
            is_f_mode = not is_f_mode

        a=show_batches_time()
        return a
    # print(main(3,[1,9,1],[1,3,3],[0,0],99,0,0))
    import heapq
    import random
    random.seed(0)
    # layer_forward_list=[random.uniform(5, 10) for _ in range(40)]
    # layer_backward_list=[random.uniform(5, 10) for _ in range(40)]
    # layer_communication_list=[random.uniform(1, 3) for _ in range(39)]


    layer_communication_list_=layer_communication_list.copy()
    layer_communication_list_.insert(0,layer_communication_list[0])
    layer_process_time=[x + y+z for x, y,z in zip(layer_forward_list, layer_backward_list,layer_communication_list_)]

    for i in range(len(layer_process_time)):
        if i==0 or i==len(layer_process_time)-1:
            continue
        layer_process_time[i]+=layer_communication_list_[i+1]
    # straggle_for_stage=[1,1,1,1] #
    # stage_num=4
    # stage_nums=[6,14,12,8] #
    layer_forward_list_new=[]
    layer_backward_list_new=[]
    layer_communication_list_new=[]
    # top_k=int(min(stage_nums)/10)
    # top_k=6  #super

    def top_k_max_indices(lst, start,end,k):
        # 使用heapq模块的nlargest函数找到前k个最大值的索引
        max_indices = heapq.nlargest(k, range(start,end), key=lst.__getitem__)
        max_indices.sort()
        return max_indices
    def calculate_stage_end_indices(stage_sizes):
        # 计算每个阶段的结束index
        end_indices = [sum(stage_sizes[:i+1]) for i in range(len(stage_sizes))]
        return end_indices
    stage_index=calculate_stage_end_indices(stage_nums)
    stage_index.insert(0,0)
    print("stage_index",stage_index)
    max_indexes=[]
    for i in range(1,len(stage_index)):
        max_index=top_k_max_indices(layer_process_time,stage_index[i-1],stage_index[i],top_k)
        max_index=[x+1 for x in max_index]
        max_indexes+=max_index
    max_indexes.insert(0,0)
    max_indexes.append(len(layer_forward_list))
    for i in range(1,len(max_indexes)-1):
        layer_forward_list_new.append(sum(layer_forward_list[max_indexes[i-1]:max_indexes[i]]))
        layer_backward_list_new.append(sum(layer_backward_list[max_indexes[i-1]:max_indexes[i]]))
    print(max_indexes)
    print(layer_forward_list_new)
    print(layer_backward_list_new)
    for i in range(1,len(max_indexes)-2):
        layer_communication_list_new.append(layer_communication_list[max_indexes[i]-1])
    if (max_indexes[-2]-1)!=len(layer_forward_list)-1:
        layer_forward_list_new[-1]+=sum(layer_forward_list[max_indexes[-2]:len(layer_forward_list)])
        layer_backward_list_new[-1]+=sum(layer_backward_list[max_indexes[-2]:len(layer_forward_list)])
    import numpy as np
    import time
    time_begin=time.time()

    #for test
    # layer_forward_list_new=layer_forward_list
    # layer_communication_list_new=layer_communication_list
    # layer_backward_list_new=layer_backward_list

    record=np.full((len(layer_forward_list_new),)*(stage_num-1),np.inf)
    # for i in range(1,len(layer_forward_list_new)):
    #     for j in range(1,len(layer_forward_list_new)):
    #         for k in range(1,len(layer_forward_list_new)):
    #             if i<j<k:
    #                 print(i,j,k)
    #                 present_stage_forward=[]
    #                 present_stage_backward=[]
    #                 present_stage_forward.append(straggle_for_stage[0]*sum(layer_forward_list_new[0:i]))
    #                 present_stage_forward.append(straggle_for_stage[1]*sum(layer_forward_list_new[i: j]))
    #                 present_stage_forward.append(straggle_for_stage[2]*sum(layer_forward_list_new[j:k]))
    #                 present_stage_forward.append(straggle_for_stage[3]*sum(layer_forward_list_new[k:len(layer_forward_list_new)]))
    #
    #                 present_stage_backward.append(straggle_for_stage[0]*sum(layer_backward_list_new[0:i]))
    #                 present_stage_backward.append(straggle_for_stage[1]*sum(layer_backward_list_new[i:j]))
    #                 present_stage_backward.append(straggle_for_stage[2]*sum(layer_backward_list_new[j:k]))
    #                 present_stage_backward.append(straggle_for_stage[3]*sum(layer_backward_list_new[k:len(layer_forward_list_new)]))
    #                 record[i][j][k]=main(stage_num,present_stage_forward,present_stage_backward,layer_communication_list_new,99,0,0)
    #                 # print(present_stage_forward)
    #                 # print(present_stage_backward)
    #             else:
    #                 continue
    for i in range(1,len(layer_forward_list_new)):

        present_stage_forward=[]
        present_stage_backward=[]
        layer_communication_list_new_=[]
        present_stage_forward.append(straggle_for_stage[0]*sum(layer_forward_list_new[0:i]))
        present_stage_forward.append(straggle_for_stage[1]*sum(layer_forward_list_new[i:len(layer_forward_list_new)]))

        present_stage_backward.append(straggle_for_stage[0]*sum(layer_backward_list_new[0:i]))
        present_stage_backward.append(straggle_for_stage[1]*sum(layer_backward_list_new[i:len(layer_forward_list_new)]))
        layer_communication_list_new_.append(layer_communication_list_new[i-1])
        # print(present_stage_forward,present_stage_backward,max_indexes[i]-1,layer_communication_list_new_)
        record[i]=main(stage_num,present_stage_forward,present_stage_backward,layer_communication_list_new_,99,0,0)
                    # print(present_stage_forward)
                    # print(present_stage_backward)
    flat_index_of_min = np.argmin(record)
    # 将扁平化索引转换为多维索引
    min_index = np.unravel_index(flat_index_of_min, record.shape)
    print("最小值的下标:", min_index)
    print("final result",record[min_index[0]])
    print("time",time.time()-time_begin)
    new_stage_nums=[]
    # new_stage_nums.append(max_indexes[min_index[0]])
    # for i in range(1,stage_num-1):
    #     new_stage_nums.append(max_indexes[min_index[i]]-max_indexes[min_index[i-1]])
    # new_stage_nums.append(len(layer_forward_list)-max_indexes[min_index[stage_num-2]])
    # new_stage_nums=[max_indexes[min_index[0]],max_indexes[min_index[1]]-max_indexes[min_index[0]],
    #                 max_indexes[min_index[2]]-max_indexes[min_index[1]],len(layer_forward_list)-max_indexes[min_index[2]]]
    new_stage_nums=[max_indexes[min_index[0]],len(layer_forward_list)-max_indexes[min_index[0]]]
    print("rearange",new_stage_nums)
    print(record)
    return new_stage_nums
import random

with open('vgg16_data_1_for_.txt', 'r') as file:
    data = file.read().splitlines()
layer_forward_list = [float(x) for x in data]
with open('vgg16_data_1_bac_.txt', 'r') as file:
    data = file.read().splitlines()
layer_backward_list = [float(x) for x in data]
with open('vgg16_out_put_com_time.txt', 'r') as file:
    data = file.read().splitlines()
layer_communication_list = [1000*float(x) for x in data]
# layer_communication_list = [0]*37
# print(layer_communication_list)
print(sum(layer_forward_list[0:10])+sum(layer_backward_list[0:10]))
print(sum(layer_forward_list[10:-1])+sum(layer_backward_list[10:-1]))
print(layer_communication_list[9])
a=calculate_new_placement(layer_forward_list=layer_forward_list,layer_backward_list=layer_backward_list,
                        layer_communication_list=layer_communication_list,straggle_for_stage=[1,1],stage_num=2,stage_nums=[19,19],
                        top_k=19)
print(a)

# def main(stage_num, forward_cost_list, backward_cost_list, comm_cost_list, max_micro_batch_num, cur_micro_batch_num,
#          timestamp):
#     # # Total stage number
#     # stage_num = 2
#     # # Forward cost and backward cost list of each stage
#     # forward_cost_list = [1, 1]
#     # backward_cost_list = [3, 3]
#     # # Communication cost list of each stage. e.g. The first element is the communication cost between stage 1 and stage 2.
#     # comm_cost_list = [1]
#     #
#     # max_micro_batch_num = 99
#     # cur_micro_batch_num = 0
#     # timestamp = 0
#
#     def get_comm_cost(stage1: float, stage2: float) -> float:
#         """计算两个相邻 stage 的通信时间 (stage 顺序不做限制)。
#
#         Args:
#             stage1 (int): 前一个 stage 的编号。
#             stage2 (int): 后一个 stage 的编号。
#
#         Returns:
#             int: _description_
#         """
#         assert abs(stage1 - stage2) == 1, "The stages must be adjacent!"
#         return comm_cost_list[min(stage1, stage2)]
#
#     class Stage:
#         def __init__(self, idx: int, forward_cost: float, backward_cost: float) -> None:
#             self.idx = idx
#             self.forward_cost = forward_cost
#             self.backward_cost = backward_cost
#
#             self.warmup_mb_num = -1  # not used yet
#
#             self.f_idx = -1  # index of next forward micro_batch
#             self.b_idx = -1  # index of next backward micro_batch
#
#         def __str__(self) -> str:
#             return f"stage_idx: {self.idx}\tforward_cost: {self.forward_cost}\tbackward_cost: {self.backward_cost}\tf_idx: {self.f_idx}\tb_idx: {self.b_idx}."
#
#         def get_next_f_idx(self) -> int:
#             assert self.b_idx >= 0, "Previous backward index is not initialized!"
#             return self.b_idx + (stage_num - self.idx)
#
#         def get_next_b_idx(self) -> int:
#             assert self.f_idx >= 0, "Previous forward index is not initialized!"
#             return self.f_idx - (stage_num - 1 - self.idx)
#
#         def get_prev_b_idx(self) -> int:
#             assert self.f_idx >= 0, "Current forward index is not initialized!"
#             return self.f_idx - (stage_num - self.idx)
#
#         def get_prev_f_idx(self) -> int:
#             assert self.b_idx >= 0, "Current backward index is not initialized!"
#             return self.b_idx + (stage_num - 1 - self.idx)
#
#     def generate_stages():
#         stages = [Stage(i, forward_cost_list[i], backward_cost_list[i]) for i in range(stage_num)]
#         return stages
#
#     stages = generate_stages()
#
#     class MicroBatch:
#         def __init__(self, stage_num: int) -> None:
#             self.stage_num = stage_num
#             self.forward_ts = -1
#             self.backward_ts = -1
#
#             self.forward_cost = stages[self.stage_num].forward_cost
#             self.backward_cost = stages[self.stage_num].backward_cost
#
#         def __str__(self) -> str:
#             return f"stage: {self.stage_num}\tforward_ts: {self.forward_ts}\tbackward_ts: {self.backward_ts}."
#
#     def generate_batches():
#         batches = [
#             [
#                 MicroBatch(stage)
#                 for _ in range(max_micro_batch_num)
#             ] \
#             for stage in range(stage_num)
#         ]
#         return batches
#
#     def get_warmup_micro_batch_num(stage: int) -> int:
#         return stage_num - stage - 1
#
#     batches = generate_batches()
#
#     def show_batches(show_diff: bool = True):
#         last_forward_ts, last_backward_ts = -1, -1
#         for stage_idx in range(stage_num):
#             print(f"===== stage {stage_idx} =====")
#             print(type(batches[stage_idx]), len(batches[stage_idx]))
#             for idx, stage in enumerate(batches[stage_idx]):
#                 if idx == len(batches[stage_idx]) - stage_num:
#                     return
#                 if show_diff:
#                     if idx % stage_num == 0:
#                         if idx != 0:
#                             print(
#                                 f"mb_index: {idx}\tforward_diff: {stage.forward_ts - last_forward_ts}\tbackward_diff: {stage.backward_ts - last_backward_ts}"
#                             )
#                         last_forward_ts = stage.forward_ts
#                         last_backward_ts = stage.backward_ts
#                 else:
#                     print(f"mb_index: {idx}", stage)
#
#     def show_batches_time(show_diff: bool = True):
#         last_forward_ts, last_backward_ts = -1, -1
#         time_sum = 0
#         count = 0
#         for stage_idx in range(1):
#             for idx, stage in enumerate(batches[stage_idx]):
#                 if idx == len(batches[stage_idx]) - stage_num:
#                     return time_sum / (stage_num * count)
#                 if show_diff:
#                     if idx % stage_num == 0:
#                         if idx != 0:
#                             count += 1
#                             time_sum += stage.forward_ts - last_forward_ts
#                         last_forward_ts = stage.forward_ts
#                 else:
#                     print(f"mb_index: {idx}", stage)
#
#     def show_stages():
#         for stage in stages:
#             print(stage)
#
#     # * 1. Warmup stage
#     for stage in range(stage_num):
#         warmup_micro_batch_num = get_warmup_micro_batch_num(stage)
#         stages[stage].warmup_mb_num = warmup_micro_batch_num
#         if stage != stage_num - 1:
#             if stage == 0:
#                 batches[stage][0].forward_ts = timestamp
#
#             for i in range(warmup_micro_batch_num):
#                 if stage == 0:
#                     if i != 0:
#                         batches[stage][i].forward_ts = \
#                             batches[stage][i - 1].forward_ts + batches[stage][i - 1].forward_cost
#                 else:
#                     batches[stage][i].forward_ts = max(
#                         batches[stage][i - 1].forward_ts + batches[stage][i - 1].forward_cost,
#                         batches[stage - 1][i].forward_ts + batches[stage - 1][i].forward_cost +
#                         get_comm_cost(stage - 1, stage)
#                     )
#             # update f_idx
#             stages[stage].f_idx = warmup_micro_batch_num
#         else:
#             # update f_idx
#             stages[stage].f_idx = 0
#
#     # show_batches()
#     # show_stages()
#
#     # * 2. Running 1F1B
#     is_f_mode = True
#     exit_flag = False
#     while cur_micro_batch_num <= max_micro_batch_num and (not exit_flag):
#         # * Forward pass
#         if is_f_mode:
#             for stage_idx in range(stage_num):
#                 stage = stages[stage_idx]
#
#                 prev_b_idx = stage.get_prev_b_idx()
#                 prev_backward_finish_ts = -1 if prev_b_idx < 0 else \
#                     batches[stage_idx][prev_b_idx].backward_ts + batches[stage_idx][prev_b_idx].backward_cost
#
#                 if stage_idx == 0:
#                     batches[stage_idx][stage.f_idx].forward_ts = max(
#                         batches[stage_idx][stage.f_idx - 1].forward_ts +
#                         batches[stage_idx][stage.f_idx - 1].forward_cost, prev_backward_finish_ts
#                     )
#                 elif stage_idx == stage_num - 1:
#                     batches[stage_idx][stage.f_idx].forward_ts = max(
#                         batches[stage_idx - 1][stage.f_idx].forward_ts +
#                         batches[stage_idx - 1][stage.f_idx].forward_cost + get_comm_cost(stage_idx - 1, stage_idx),
#                         prev_backward_finish_ts
#                     )
#                 else:
#                     batches[stage_idx][stage.f_idx].forward_ts = max(
#                         prev_backward_finish_ts,
#                         max(
#                             batches[stage_idx][stage.f_idx - 1].forward_ts +
#                             batches[stage_idx][stage.f_idx - 1].forward_cost,
#                             batches[stage_idx - 1][stage.f_idx].forward_ts +
#                             batches[stage_idx - 1][stage.f_idx].forward_cost + get_comm_cost(stage_idx - 1, stage_idx)
#                         )
#                     )
#                 # update cur_micro_batch_num
#                 cur_micro_batch_num = max(cur_micro_batch_num, stage.f_idx)
#                 # update b_idx
#                 next_b_idx = stage.get_next_b_idx()
#                 # print(f"next_b_idx: {next_b_idx}")
#                 stage.b_idx = next_b_idx
#                 if stage.b_idx >= max_micro_batch_num:
#                     exit_flag = True
#                     break
#
#         # * Backward pass
#         else:
#             for stage_idx in range(stage_num - 1, -1, -1):
#                 stage = stages[stage_idx]
#
#                 prev_f_idx = stage.get_prev_f_idx()
#                 prev_forward_finish_ts = -1 if prev_f_idx < 0 else \
#                     batches[stage_idx][prev_f_idx].forward_ts + batches[stage_idx][prev_f_idx].forward_cost
#
#                 if stage_idx == 0:
#                     batches[stage_idx][stage.b_idx].backward_ts = max(
#                         prev_forward_finish_ts, \
#                         batches[stage_idx + 1][stage.b_idx].backward_ts + batches[stage_idx + 1][
#                             stage.b_idx].backward_cost + get_comm_cost(stage_idx, stage_idx + 1)
#                     )
#                 elif stage_idx == stage_num - 1:
#                     batches[stage_idx][stage.b_idx].backward_ts = prev_forward_finish_ts
#                 else:
#                     batches[stage_idx][stage.b_idx].backward_ts = max(
#                         prev_forward_finish_ts, \
#                         batches[stage_idx + 1][stage.b_idx].backward_ts + batches[stage_idx + 1][
#                             stage.b_idx].backward_cost + get_comm_cost(stage_idx, stage_idx + 1)
#                     )
#                 # update cur_micro_batch_num
#                 cur_micro_batch_num = max(cur_micro_batch_num, stage.b_idx)
#                 # update f_idx
#                 next_f_idx = stage.get_next_f_idx()
#                 # print(f"prev_f_idx: {stage.f_idx}, next_f_idx: {next_f_idx}")
#                 stage.f_idx = next_f_idx
#                 if stage.f_idx >= max_micro_batch_num:
#                     exit_flag = True
#                     break
#
#         # change mode
#         is_f_mode = not is_f_mode
#
#     a = show_batches_time()
#     return a
# print(main(2,[300,25],[300,25],[396],99,0,0))