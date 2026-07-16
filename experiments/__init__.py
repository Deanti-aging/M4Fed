# experiments 子包：M4Fed 所有实验脚本的集合
# 各模块说明：
#   run_all_baselines : 运行全部 5 个联邦基线并生成对比结果图表
#   ablation          : 消融实验（验证公开数据/TabPFN/预训练骨干各组件的贡献）
#   sensitivity       : 超参数敏感性分析（合成比例/学习率/Dirichlet α 网格搜索）
#   cross_site        : 跨站点泛化实验（在未见过的采集中心数据上评估服务器 SwinMultiTask）
# 该包无需暴露公开 API，各模块通过命令行直接运行或被 main.py 按需导入
