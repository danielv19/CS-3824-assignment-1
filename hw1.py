import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv

def input_to_df(name):
    nodes_df = pd.read_csv(f"inputs/{name}-nodes.txt", sep='\t')
    edges_df = pd.read_csv(f"inputs/{name}-edges.txt", sep='\t')
    nodes_final = nodes_df.drop('node_symbol',axis=1)
    edges_final = edges_df[['#tail','head','weight','edge_type']]
    return [nodes_final, edges_final]

def edge_exists(str, edge_dict, simple):
    edges = str.split(" ")
    if f"{edges[0]}{edges[1]}" in edge_dict:
        return f"{edges[0]} {edges[1]}"
    elif not simple and f"{edges[1]}{edges[0]}" in edge_dict:
        return f"{edges[1]} {edges[0]}"
    return False

def graph_edge_check(tail, head, graph):
    return (tail, head) in graph.edges() or (head, tail) in graph.edges() 

#input datasets
def graph_making(pathway):
    data = input_to_df(pathway)
    #nodes
    temp = data[0].drop(data[0][data[0]['node_type'] == 'none'].index)
    receptor_df = temp.drop(temp[temp['node_type'] == 'tf'].index)
    tf_df = temp.drop(temp[temp['node_type'] == 'receptor'].index)
    tfs = []
    receptors = []
    [tfs.append(x) for x in tf_df['#node'] if x not in tfs]
    [receptors.append(x) for x in receptor_df['#node'] if x not in receptors]
    tf_rcp = {}
    for x in tfs:
        tf_rcp[x] = 'tf'
    for x in receptors:
        tf_rcp[x] = 'receptor'

    #edges
    tails = []
    [tails.append(x) for x in data[0]['#node'] if x not in tails]
    all_connections = pd.read_csv("inputs/pathlinker-human-network.txt", sep='\t')
    all_final = all_connections.drop('edge_type',axis=1)
    all_final["code"] = all_final["#tail"] + all_final["head"]
    human_network = all_final["code"].to_numpy()

    g = nx.DiGraph()
    g.add_nodes_from(tails)
    edge_dict = {}
    for x in range(len(data[1])):
        tail = data[1].iloc[x]['#tail']
        head = data[1].iloc[x]['head']
        weight = data[1].iloc[x]['weight']
        if not edge_exists(f"{head} {tail}", human_network, False):
            print(F"DENIED {pathway} {tail}\t{head}")
        if not edge_exists(f"{head} {tail}", edge_dict, True):
            if data[1].iloc[x]['edge_type'] != 'physical':
                g.add_edge(tail, head, weight=weight)
                edge_dict[f"{tail}{head}"] = 1
            else:
                g.add_edge(tail, head, weight=weight)
                g.add_edge(head, tail, weight=weight)
                edge_dict[f"{tail}{head}"] = 2
        elif isinstance(edge_exists(f"{head} {tail}", edge_dict, False),str) and data[1].iloc[x]['edge_type'] == 'physical':
                string = edge_exists(f"{head} {tail}", edge_dict, False)
                edges = string.split(" ")
                g.add_edge(edges[1], edges[0])
                edge_dict[string] = 2
    return [g, tfs, receptors]

#input datasets
def precision_recall_calc(data, graph):
    recall_denom = len(graph.edges())
    precision_denom = 1
    numerator = 1 if (graph_edge_check(data.iloc[0]["#tail"], data.iloc[0]['head'], graph)) else 0
    recall = []
    precision = []
    precision.append(numerator/precision_denom)
    recall.append(numerator/recall_denom)
    for x in range(1, len(data)):
        precision_denom += 1
        row = data.iloc[x]
        before = data.iloc[x-1]['KSP index']
        numerator = numerator + 1 if (graph_edge_check(row["#tail"], row['head'], graph)) else numerator
        if row['KSP index'] == before:
            recall[len(recall)-1] = numerator/recall_denom
            precision[len(recall)-1] = numerator/precision_denom
        else:
            recall.append(numerator/recall_denom)
            precision.append(numerator/precision_denom)

    return [precision, recall]

def pathlinker_precision_recall_plot():
    plt.title("PathLinker Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    pathways = ["Wnt", "TNFalpha", "TGF_beta_Receptor", "EGFR1" ]
    colors = ["red", "blue", "green", "yellow"]
    for x in range(4):
        gra = graph_making(pathways[x])[0]
        ranked_df = pd.read_csv(f"results/{pathways[x]}-k_20000-ranked-edges.txt", sep='\t')
        data = precision_recall_calc(ranked_df, gra)
        plt.plot(data[1], data[0], color=colors[x])
    plt.legend(pathways)
    plt.show()

def get_weight_edge(u,v,dict):
    return dict['weight']

def format_edges(lis, edges, rank_a, rank):
    for nodes in lis:
        for x in range(len(nodes)-1):
            if f"{nodes[x]}\t{nodes[x+1]}" not in edges:
                edges.append(f"{nodes[x]}\t{nodes[x+1]}")
                rank_a.append(rank)
    return [edges, rank_a, rank]

def shortest_path_algorithm():
    all_connections = pd.read_csv("inputs/pathlinker-human-network.txt", sep='\t')
    all_final = all_connections.drop('edge_type',axis=1)
    all_final["code"] = all_final["#tail"] + all_final["head"]
    gi = nx.DiGraph()
    for x in range(len(all_final)):
        tail = all_final.iloc[x]['#tail']
        head = all_final.iloc[x]['head']
        weight = all_final.iloc[x]['neglog10edge_weight']
        gi.add_edge(tail, head, weight=weight)


    pathway = ["Wnt", "TNFalpha", "TGF_beta_Receptor", "EGFR1" ]
    for j in range(4):
        rank = 1
        edges = []
        ranks = []
        g = graph_making(pathway[j])
        for x in g[2]:
            for y in g[1]:
                try:
                    length = nx.all_shortest_paths(gi,x,y, weight=get_weight_edge, method="dijkstra")
                    length = [path for path in length]
                    temp = format_edges(length, edges, ranks, rank)
                    edges = temp[0]
                    ranks = temp[1]
                    rank += 1
                except:
                    pass 

        f = open(f"spa output/{pathway[j]}_spa.txt", "w")
        f.write("#tail\thead\tKSP index\n")
        for x in range(len(edges)):
            f.write(f"{edges[x]}\t{ranks[x]}\n")
        f.close()

#pathlinker_precision_recall_plot()

def p(pathway,g, q):
    #load in human network
    edge_dict = {key: val for val, key in enumerate(list(g.nodes))}
    out_degree = {}
    costs = {}
    #create sparse representation of A
    row = []
    col = []
    data = []
    for u,v,weight in g.edges(data=True):
        row.append(edge_dict.get(u))
        col.append(edge_dict.get(v))
        data.append(1)
        if u not in costs:
            costs[u] = 1
        else:
            costs[u] += 1
        if u not in out_degree:
            out_degree[u] = weight['weight']
        else:
            out_degree[u] += weight['weight']
    
    #creating sparse matrix for A
    sparse_a = csr_matrix((data, (row, col)),shape=(len(list(g.nodes)),len(list(g.nodes))))

    #creating I indentity matrix
    identity = [1] * len(list(g.nodes))
    identity_index = list(range(0, len(g.nodes)))
    sparse_i = csr_matrix((identity, (identity_index, identity_index)),shape=(len(list(g.nodes)),len(list(g.nodes))))

    #creating inverse D matrix
    cost = []
    row_d = []
    for x in costs:
        row_d.append(edge_dict[x])
        cost.append(float(1)/float(costs[x]))
    sparse_d_inv = csr_matrix((cost, (row_d, row_d)),shape=(len(list(g.nodes)),len(list(g.nodes))))

    temp = pathway.drop(pathway[pathway['node_type'] == 'none'].index)
    receptor_df = temp.drop(temp[temp['node_type'] == 'tf'].index)
    receptors = []
    [receptors.append(x) for x in receptor_df['#node'] if x not in receptors]
    s_data = []
    s_row = []
    s_col = [0] * len(list(g.nodes))
    for x in list(g.nodes()):
        if x in receptors:
            s_data.append(float(1)/float(len(receptors)))
        else:
            s_data.append(0)
        s_row.append(edge_dict.get(x))
    sparse_s = csr_matrix((s_data, (s_row, s_col)),shape=(len(list(g.nodes)),1))

    #D^-1 * A
    calculation_so_far = sparse_d_inv.tocsc() @ sparse_a.tocsc()
    #(D^-1 * A)^T
    transposed = calculation_so_far.transpose()
    #(1-q)(D^-1 * A)^T
    transposed = (1-q)*transposed
    #I - (1-q)(D^-1 * A)^T
    equation = sparse_i.tocsc() - transposed.tocsc()
    #(I - (1-q)(D^-1 * A)^T)^-1
    inv_equation = inv(equation)
    #q*s
    qs = q*sparse_s
    #(I - (1-q)(D^-1 * A)^T)^-1 * qs
    inv_equation = inv_equation@qs
    return [inv_equation,out_degree]


def flux_outputs(pathway):
    #input pathway
    data = input_to_df(pathway)

    human_network = pd.read_csv("inputs/pathlinker-human-network.txt", sep='\t')
    human_network.columns = ['Tail', 'head', 'weight', 'Type']
    edge_dict = {key: val for val, key in enumerate(list(g.nodes))}
    human_g = nx.DiGraph()
    for x in range(len(human_network)):
        tail = human_network.iloc[x]['Tail']
        head = human_network.iloc[x]['head']
        weight = human_network.iloc[x]['weight']
        human_g.add_edge(tail, head, weight=weight)
    subgraph_nodes = max(nx.strongly_connected_components(human_g), key=len)
    human_g = human_g.subgraph(subgraph_nodes)
    p_val = p(data[0], human_g,0.5)

    f = open(f"spa output/{pathway[j]}_spa.txt", "w")
    f.write("#tail\thead\tedge_flux\n")
    for u,v,weight in human_g.edges(data=True):
        #p(u) * w(u,v) / d_out(u)
        flux = (p_val[0][edge_dict.get(u)].toarray()[0][0]*weight['weight'])/(p_val[1].get(u))
        f.write(f"{u}\t{v}\t{flux}\n")
    f.close()

def shortest_path_vs_pathlinker_pathway(pathway):
    plt.title(f"{pathway} Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    colors = ["red", "blue"]
    for x in range(2):
        gra = graph_making(pathway)[0]
        if x == 0:
            ranked_df = pd.read_csv(f"results/{pathway}-k_20000-ranked-edges.txt", sep='\t')
        else:
            ranked_df = pd.read_csv(f"spa output/{pathway}_spa.txt", sep='\t')
        data = precision_recall_calc(ranked_df, gra)
        plt.plot(data[1], data[0], color=colors[x])
    plt.legend(["PathLinker","RWR"])
    plt.show()

def shortest_path_vs_pathlinker():
    pathways = ["Wnt", "TNFalpha", "TGF_beta_Receptor", "EGFR1" ]
    for path in pathways:
        shortest_path_vs_pathlinker_pathway(path)
