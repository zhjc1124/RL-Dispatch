# RL-Dispatch

包含数据为2021-11-01~2021-11-30的数据，共30天的数据  
包裹数据包含30天的数据，地铁数据只包含一天的数据，这里每天重复使用地铁数据。

### 关键函数

env.reset(day=1)  
参数day表明为哪一天，默认为day=1  

env.step(actions)
传入每一个包裹对应的action，即包裹下一站发往哪  

上述两个函数均返回state状态，为一个dict数组，包含键为：  
dispatchs：需要派送的包裹，格式为list[int, int, int]，包含值[当前站点，目标站点，剩余时间限制]  
packages：见论文Demand states，格式为list[1x118的torch数组，118x118的torch数组]，包含值为[各站点的等待包裹，当前正在运输的包裹]  
passengers：见论文Supply states，格式为list[1x118的torch数组，118x118的torch数组]，包含值为[各站点的入站乘客，在途中的乘客]
done：是否结束，True或者False    
time：时间  
reward：该时刻获得的reward值
