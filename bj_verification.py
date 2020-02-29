# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:33:47 2020

@author: user
"""
import blackjack as bj
import numpy as np



parameters = np.array([-0.347975	,-0.880453	,0.322294	,0.0125656	,0.343709	,0.260684	,-0.0269152	,-0.0494893	,0.0377596	,0.861108	,0.865142	,-0.388869	,0.778311	,0.952508	,0.29716	,0.943587	,0.909729	,1	,-0.972446	,-0.0476327	,0.624991	,-0.477773	,-0.253361	,1	,-0.93443	,-0.341684	,-0.818403	,0.943215	,-0.151897	,0.87357	,-0.733477	,-0.0983403	,-0.0704627	,0.695232	,0.0483744	,-0.0615406	,0.977257	,-0.727561	,0.535447	,-0.354503	,-0.0883583	,-0.19138	,-0.822418	,-0.357621	,-0.5403	,-0.940529	,-0.121779	,-0.830636	,0.083038	,-0.0794957	,-0.370195	,-0.939474	,-0.244791	,0.464646	,0.153237	,-1	,-0.46393	,-0.673006	,0.279632	,-0.840913	,-0.0322773	,-0.568079	,-0.690632	,0.603158	,0.964596	,0.3324	,-0.665454	,0.610517	,-0.827189	,-1	,-0.107342	,-0.642	,-0.480853	,-0.251165	,-0.149296	,0.374908	,-1	,-0.674375	,0.549571	,0.0580538	,0.965694	,-0.524754	,-0.0598357	,-0.329005	,-0.194974	,-0.105485	,0.426377	,0.320325	,0.129861	,0.501759	,0.441746	,0.470813	,0.517485	,-1	,0.240293	,-0.576208	,-0.873817	,-0.810497	,-0.300709	,-0.437081	,-0.495659	,-0.715943	,-0.0760045	,0.26753	,-0.474437	,-0.994447	,-0.0677103	,0.604761	,0.867912	,-0.209684	,1	,0.95238	,0.0774753	,0.754941	,0.33101	,-0.54648	,-0.0433487	,0.0159289	,0.707203	,0.774987	,0.019283	,-0.567998	,-0.647147	,0.921974	,0.626372	,0.078511	,0.636215	,0.202642	,0.089843	,-0.162821	,-0.572548	,-0.67519	,0.476743	,-0.840358	,-0.863853	,-0.119655	,0.812718	,-0.607672	,-0.622625	,0.138205	,0.451854	,0.954013	,-0.090575	,0.556312	,-0.194499	,0.831195	,-1	,-0.844733	,1	,-0.637698	,0.50245	,0.473106	,0.497564	,-0.136426	,0.377525	,-0.622711	,0.660624	,0.647832	,0.135641	,-0.818811	,0.353205	,-0.830369	,-0.802647	,-0.538863	,-0.187892	,-0.0543038	,0.0964166	,-0.181726	,-0.9193	,0.140658	,0.874566	,0.814336	,-0.42998	,0.137508	,0.663697	,0.0124463	,-0.76055	,0.651687	,0.646134	,0.725064	,-0.436836	,-0.258608	,0.837316	,-0.81762	,-0.271755	,0.42369	,0.279713	,-0.802567	,-0.352724	,0.121225	,-0.824461	,-0.745659	,-0.0177015	,0.55678	,-0.541817	,-0.187288	,-0.777733	,-0.897186	,-1	,-0.201491	,-1	,-0.317461	,-0.638262	,-0.807585	,0.369598	,0.308619	,0.231826	,0.774192	,-0.217471	,-0.386999	,0.233814	,-0.426853	,-0.455248	,0.0947177	,-0.955992	,-0.186988	,1	,0.111177	,0.539053	,-0.872254	,0.119918	,0.383478])
parameters_rand = np.random.uniform(-1,1,222)
#bj.game(parameters)
print(bj.win_mean(parameters,100000))
