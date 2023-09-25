# For now this collection of functions is split into two classes:

# 'ThoughtGraph' should be instantiated as an object (some day the node and edge tables will be stored in this object and this will make more sense), for which you will normally only call one method:
#   from ThoughtGraph import ThoughtGraph
#   tgraph = ThoughtGraph()
#   tgraph.export_to_vis_js(nodes_df, edges_df, title, html_file_name)

# The 'utils' class is just a collection of class methods, and it should not be instantiated:
#   from ThoughtGraph import utils
#   sentences = ['The cat sat on a mat.', 'The dog chewed on a log.']
#   part_of_speech_vector = utils.text_to_pos(sentences)

class ThoughtGraph():

    def __init__(self, nodes_df, edges_df):
        self.nodes = nodes_df
        self.edges = edges_df
    
    def get_vis_js_html(self):
        """
        Generate HTML encoding vis_js graph from Pandas dataframes of nodes and edges.
        """
        
        nodes_str = self.nodes.to_json(orient='records')
        edges_str = self.edges.to_json(orient='records')
        
        max_weight = max(self.edges['weight'])
    
        html_string = ( 
        '     <style type="text/css">#mynetwork {width: 100%; height: 1000px; border: 3px}</style>\n'
        '     <button onclick=toggle_motion()>Toggle motion</button>\n'
        '     <div class="slidercontainer">\n'
        '            <label>minimum edge weight:\n'
        f'                <input type="range" min="0" max="{max_weight}" value="{max_weight/2}" step="{max_weight/100}" class="slider" id="min_edge_weight">\n'
        '                <input type="text" id="min_edge_weight_display" size="2">\n'
        '            </label>\n'
        '     </div>\n'
        '     <div id="mynetwork"></div>\n'
        f'     <script type="text/javascript">NODE_LIST={nodes_str};FULL_EDGE_LIST={edges_str};</script>\n'
        '     <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>\n'
        '     <script type="text/javascript">\n'
        '       const sign_color = {pos:"blue", neg:"red", zero:"black"}\n'
        '       const options = {physics:{maxVelocity: 1, minVelocity: 0.01}}\n'
        '       var edgeFilterSlider\n'
        '       var mynetwork\n'
        '       var motion_flag = false\n'
        '       function toggle_motion(){\n'
        '           motion_flag = !motion_flag\n'
        '           mynetwork.setOptions( { physics: motion_flag } )\n'
        '       }\n'
        '       function edgesFilter(edge){ return edge.value >= edgeFilterSlider.value }\n'
        '       function init_network(){\n'
        '           document.getElementById("min_edge_weight_display").value = 0.5\n'
        '           document.getElementById("min_edge_weight").onchange = function(){\n'
        '               document.getElementById("min_edge_weight_display").value = this.value\n'
        '           }\n'
        '           edgeFilterSlider = document.getElementById("min_edge_weight")\n'
        '           edgeFilterSlider.addEventListener("change", (e) => {edgesView.refresh()})\n'
        '           var container = document.getElementById("mynetwork")\n'
        '           var EDGE_LIST = []\n'
        '           for (var i = 0; i < FULL_EDGE_LIST.length; i++) {\n'
        '               var edge = FULL_EDGE_LIST[i]\n'
        '               edge["value"] = Math.abs(edge["weight"])\n'
        '               edge["title"] = "weight " + edge["weight"]\n'
        '               edge["sign"] = (edge["weight"] < 0) ? "neg" : "pos";\n'
        '               edge["color"] = {color: sign_color[edge["sign"]] };\n'
        '               edge["arrows"] = "to"\n'
        '               EDGE_LIST.push(edge)\n'
        '           }\n'
        '           var nodes = new vis.DataSet(NODE_LIST)\n'
        '           var edges = new vis.DataSet(EDGE_LIST)\n'
        '           var nodesView = new vis.DataView(nodes)\n'
        '           var edgesView = new vis.DataView(edges, { filter: edgesFilter })\n'
        '           var data = { nodes: nodesView, edges: edgesView }\n'
        '           mynetwork = new vis.Network(container, data, options)\n'
        '       }\n'
        '       init_network()\n'
        '     </script>\n'
    
        )
        return html_string
    
    
    def export_to_vis_js(self, title, html_file_name):
        """
        Generate vis_js graph from Pandas dataframes of nodes and edges, and write to HTML file.
        """

        vis_js_html = self.get_vis_js_html()
        page_html =  ('<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '    <head>\n'
            f'       <title>{title}</title>\n'
            '    </head>\n'
            '    <body onload=init_network()>\n'
            f'{vis_js_html}'
            '\n'
            '    </body>\n'
            '</html>\n')
        
        with open(html_file_name, "wt") as html_file: 
            html_file.write(page_html)


    # Experimental visualizations: use at your own risk!

    def export_edge_thresholds_graph_to_vis_js(self, title, html_file_name):
        """
        Generate vis_js graph from cluster data using between-centroid distance as edge weights, and write to HTML file.

        Example:
          nodes_df['group'] = [1 if l.startswith('lab_') else 2 if l.startswith('d') else 0 for l in nodes_df['label']]
          tgraph = ThoughtGraph(nodes_df, edges_df)
          tgraph.export_edge_thresholds_graph_to_vis_js('Cluster Graph', 'cluster_graph.html')
        """
        max_weight = 1.0 # np.quantile(edges_df['weight'], 0.95)

        nodes_str = self.nodes.to_json(orient='records')
        edges_str = self.edges.to_json(orient='records')

        html_string = ( 
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '<head>\n'
            '	<meta http-equiv="content-type" content="text/html; charset=utf-8" />\n'
            f'	<title>{title}</title>\n'
            '	<script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>\n'
            f'	<script type="text/javascript">NODE_LIST={nodes_str}</script>\n'
            f'	<script type="text/javascript">EDGE_LIST={edges_str}</script>\n'
            '	<style type="text/css">#mynetwork {width: 100%; height: 700px; border: 1px}</style>\n'
            '	</head>\n'
            '		<body>\n'
            '			<div class="slidercontainer">\n'
            '				<label>minimum edge strength:\n'
            '					<input type="range" min="0" max="1" value="0.5" step="0.01" class="slider" id="min_edge_weight">\n'
            '					<input type="text" id="min_edge_weight_display" size="2">\n'
            '				</label>\n'
            '			</div>\n'
            '			<div id="mynetwork"></div>\n'
            '			<script type="text/javascript">\n'
            f'	const max_weight = {max_weight}\n'
            '	const weight_metric = "weight"\n'
            '	const filter_coef = {"confidence":1, "lift": max_weight, "log2lift": Math.log2(max_weight)}\n'
            '	\n'
            '	document.getElementById("min_edge_weight_display").value = max_weight * 0.5\n'
            '	document.getElementById("min_edge_weight").onchange = function(){\n'
            '		document.getElementById("min_edge_weight_display").value = max_weight * this.value\n'
            '	}\n'
            '	for (var i = 0; i < EDGE_LIST.length; i++) {\n'
            '		EDGE_LIST[i]["value"] = Math.abs(EDGE_LIST[i][weight_metric])\n'
            '	}\n'
            '	\n'
            '	const edgeFilterSlider = document.getElementById("min_edge_weight")\n'
            '	\n'
            '	function edgesFilter(edge){return edge.value > edgeFilterSlider.value * max_weight}\n'
            '	\n'
            '	const nodes = new vis.DataSet(NODE_LIST)\n'
            '	const edges = new vis.DataSet(EDGE_LIST)\n'
            '	\n'
            '	const nodesView = new vis.DataView(nodes)\n'
            '	const edgesView = new vis.DataView(edges, { filter: edgesFilter })\n'
            '	\n'
            '	edgeFilterSlider.addEventListener("change", (e) => {edgesView.refresh()})\n'
            '	\n'
            '	const container = document.getElementById("mynetwork")\n'
            '	const options = {physics:{maxVelocity: 50, minVelocity: 0.5}, edges:{arrows: "to"}}\n'
            '	const data = { nodes: nodesView, edges: edgesView }\n'
            '	new vis.Network(container, data, options)\n'
            '	\n'
            '			</script>\n'
            '		</body>\n'
            '	</html>\n'
        )
        with open(html_file_name, "wt") as html_file:
            html_file.write(html_string)


    def collapsible_clusters(self, clusters_df, title, html_file_name):
        """
        Generate interactive graph with one level of collapsible clusters.
        
        nodes_df: label, title, shape, cid (but NOT color)
        edges_df: from, to, strength (I think - maybe weight?)
        clusters_df: cid, color (must be unique for each cluster), label, title
        Note that colors must be assigned to the clusters in the 'clusters_df' dataframe; these color names are what the nodes will be grouped by.
        The consituent nodes will be colored by the clusters they belong to.
        
        tgraph = ThoughtGraph(nodes_df, edges_df)
        tgraph.collapsible_clusters(self, clusters_df, title, html_file_name)
        """
        
        nodes_str = nodes_df.to_json(orient='records')
        edges_str = edges_df.to_json(orient='records')
        clusters_str = clusters_df.to_json(orient='records')

        html_string = (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '	<!-- https://visjs.github.io/vis-network/examples/network/other/clustering.html -->\n'
            '	<head>\n'
            f'		<title>{title}</title>\n'
            '		<meta charset="utf-8"/>\n'
            '		<script\n'
            '			type="text/javascript"\n'
            '			src="https://visjs.github.io/vis-network/standalone/umd/vis-network.min.js"\n'
            '		></script>\n'
            '\n'
            '		<style type="text/css">\n'
            '			#mynetwork {\n'
            '				width: 100%;\n'
            '				height: 700px;\n'
            '				border: 1px solid lightgray;\n'
            '			}\n'
            '		</style>\n'
            '	</head>\n'
            '\n'
            '	<body>\n'
            '	<p>\n'
            '		Click on a cluster to open it. Click the "collapse clusters" button to show the clusters as single nodes.\n'
            '	</p>\n'
            '	<input type="button" onclick="collapseClusters()" value="collapse clusters" />\n'
            '	<br />\n'
            '	<div id="mynetwork"></div>\n'
            '\n'
            f'	<script type="text/javascript">nodes = {nodes_str}</script>\n'
            f'	<script type="text/javascript">edges = {edges_str}</script>\n'
            f'	<script type="text/javascript">clusters = {clusters_str}</script>\n'
            '		\n'
            '	<script type="text/javascript">\n'
            '		// color nodes by cluster\n'
            '		for (var i = 0; i < nodes.length; i++) {\n'
            '			nodes[i]["color"] = clusters[nodes[i]["cid"] - 1]["color"]\n'
            '		}\n'
            '\n'
            '		// create a network\n'
            '		var container = document.getElementById("mynetwork");\n'
            '		var data = { nodes: nodes, edges: edges };\n'
            '		var options = {\n'
            '			layout: {randomSeed: 1}, \n'
            '			physics:{maxVelocity: 50.0, minVelocity: 0.1} \n'
            '		};\n'
            '		var network = new vis.Network(container, data, options);\n'
            '		network.on("selectNode", function (params) {\n'
            '			if (params.nodes.length == 1) {\n'
            '				if (network.isCluster(params.nodes[0]) == true) {\n'
            '					network.openCluster(params.nodes[0]);\n'
            '				}\n'
            '			}\n'
            '		});\n'
            '\n'
            '		function collapseClusters() {\n'
            '			network.setData(data);\n'
            '			var clusterOptionsByData;\n'
            '			for (var i = 0; i < clusters.length; i++) {\n'
            '				var color = clusters[i]["color"];\n'
            '				clusterOptionsByData = {\n'
            '					joinCondition: function (childOptions) {\n'
            '						return childOptions.color.background == color;\n'
            '					},\n'
            '					clusterNodeProperties: {\n'
            '						id: "cluster_" + clusters[i]["cid"], // must be a string\n'
            '						borderWidth: 3,\n'
            '						shape: "database",\n'
            '						color: color,\n'
            '						label: clusters[i]["label"],\n'
            '						title: clusters[i]["title"],\n'
            '					},\n'
            '				};\n'
            '				network.cluster(clusterOptionsByData);\n'
            '			}\n'
            '		}\n'
            '\n'
            '	</script>\n'
            '	</body>\n'
            '</html>\n'
        )
        with open(html_file_name, "wt") as html_file:
            html_file.write(html_string)
            



# Collection of miscellaneous functions to be used as class methods.
# Do not create an instance of this class.
class utils():
    import os, sys, pickle, regex
    import pandas as pd
    import numpy as np
    
    def text_to_pos(text_column, parsify_attributes=['pos_', 'dep_']):
        import spacy as sp
        nlp = sp.load('en_core_web_trf')
    
        def parsify(processed, attributes=parsify_attributes):
            ptokens = []
            for token in processed:
                parts = set()
                for attrib in attributes:
                    val = token.__getattribute__(attrib).lower()
                    parts.add(val)
                ptokens.append('_'.join(parts))
            return ' '.join(ptokens)
     
        return [parsify(processed)
                for processed in nlp.pipe(text_column, n_process=12, batch_size=1000)]
    
    
    def get_item_pair_stats(item_pair_df):
        # item_pair_df must have columns named 'basket', and 'item'.
        import pandas as pd
        import sqlite3
    
        db = sqlite3.connect(":memory:")
        
        item_pair_df.to_sql("basket_item", db, if_exists="replace")
    
    
        ITEM_PAIR_STATS_QUERY = """with 
          bi as (
            select basket, item
              from basket_item
              group by basket, item  -- be sure we only count one of each kind of item per basket
          ),
          item_counts as (
            select item, count(*) item_count -- same as the number of baskets containing this item (see above)
              from bi
              group by item
          ),
          bi_count as (
            select bi.*, ic.item_count  -- basket, item, item_count
              from bi
                join item_counts ic on bi.item=ic.item
          ),
          ips as (
              select bi1.item item1, bi2.item item2,
                      bi1.item_count item1_count, bi2.item_count item2_count,
                      count(*) as both_count              
                  from bi_count bi1
                    join bi_count bi2  -- joining the table to itself
                      on bi1.basket = bi2.basket  -- two items in the same basket
                      and bi1.item != bi2.item    -- don't count the item being in the basket with itself
                  group by bi1.item, bi1.item_count, 
                           bi2.item, bi2.item_count
          ),
          cc as (
            SELECT item1, item2, item1_count, item2_count, both_count,
                  CAST(item1_count AS FLOAT)/(select count(distinct basket) from basket_item) as item1_prevalence, -- fraction of all baskets with item1
                  CAST(item2_count AS FLOAT)/(select count(distinct basket) from basket_item) as item2_prevalence, -- fraction of all baskets with item2
                  CAST(both_count AS FLOAT)/CAST(item1_count AS FLOAT) AS confidence  -- fraction of baskets with item1 that also have item2
              FROM ips
          )
        select *, confidence/item2_prevalence lift from cc
        """
    
        return pd.read_sql_query(ITEM_PAIR_STATS_QUERY, db)
    
    
    def get_nodes_and_edges_from_item_pair_stats(cooccurrence_pdf):
        """
        Convert a Pandas dataframe of item-pair statistics to separate dataframes for nodes and edges.
        """
        import pandas as pd
        from collections import Counter
        
        item_stats = {r['item1']:{'count':r['item1_count'], 'prevalence':r['item1_prevalence']} 
                        for idx, r in cooccurrence_pdf.iterrows()}
     
        item_stats.update({r['item2']:{'count':r['item2_count'], 'prevalence':r['item2_prevalence']} 
                        for idx, r in cooccurrence_pdf.iterrows()})
     
        nodes_df = pd.DataFrame([{'label':k,'count':v['count'], 'prevalence':v['prevalence']}  
                        for k,v in item_stats.items()])
        nodes_df['id'] = nodes_df.index
       
        edges_df = cooccurrence_pdf.copy()
        node_id = {r['label']:r['id'] for idx, r in nodes_df.iterrows()}
        edges_df['from'] = [node_id[nn] for nn in edges_df['item1']]
        edges_df['to'] = [node_id[nn] for nn in edges_df['item2']]
        
        print("Your graph will have {0} nodes and {1} edges.".format( len(nodes_df), len(edges_df) ))
     
        return nodes_df, edges_df[[ 'from', 'to', 'both_count', 'confidence', 'lift']]
    
    
    def make_cluster_node_title(row, text_df):
        title = f"{row['label']}\n({row['type']}, {row['count']} examples)"
        if row['type'] == 'instruction_cluster':
            cluster_id = row['label']
            examples = text_df[ text_df['instruction_B'] == cluster_id ]['instruction'].sample(6).values
            title += '\n' + '\n'.join(examples)
        if row['type'] == 'response_cluster':
            cluster_id = row['label']
            examples = text_df[ text_df['response_B'] == cluster_id ]['response'].sample(6).values
            title += '\n' + '\n'.join(examples)        
        return title
    
    
    def pivot_term_document_matrix_to_basket_item(tdm_pdf):
        import pandas as pd
        # This is just a simple pivot
        basket_item_rows = []
        for i, row in enumerate(tdm_pdf.to_dict(orient="records")):
            for k, v in row.items():
                if v > 0:
                    basket_item_rows.append({'basket': i, "item": k})
        
        basket_item = pd.DataFrame(basket_item_rows)
        return basket_item
    
    
    def get_leiden_partition(edges):
        import leidenalg   # https://pypi.org/project/leidenalg/
        import igraph as ig
        
        edge_tuple_list = [(row['from'], row['to'], row['weight']) for row in edges[['from', 'to', 'weight']].to_dict(orient='records') ]
        G = ig.Graph.TupleList(edge_tuple_list)
        
        # G = ig.Graph.DictList( edges[['from', 'to', 'weight']].to_dict(orient='records'))
        
        leiden_partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition);
        return leiden_partition.membership
    
    
    def add_cluster_cols(df, embedding_col='embedding', prefix='cluster', letters='ABCDE', max_threshold=1):
        from scipy.cluster.hierarchy import ward, fcluster
        from scipy.spatial.distance import pdist
        import math
    
        # cluster the sentence vectors at various levels
        X = df[embedding_col].tolist()
        y = pdist(X, metric='cosine')
        z = ward(y)
    
        for i in range(len(letters)):
            letter = letters[i]
            col_name = f'{prefix}_{letter}'
            cluster_id = fcluster(z, max_threshold/2**i, criterion='distance')
            digits = 1 + math.floor(math.log10(max(cluster_id)))
            df[col_name] = [col_name + str(cid).zfill(digits) for cid in cluster_id]
    
        cluster_cols = [c for c in df.columns if c.startswith(f'{prefix}_')]
        return df.sort_values(by=cluster_cols)


    def count_clusters_at_all_levels(dsc):
        cluster_cols = [c for c in dsc.columns if c.startswith("cluster_")]
        for col in cluster_cols:
            num_clusters = len(set(dsc[col]))
            print(f"{col}: {num_clusters}")


    def filter_cluster_paths(df, cluster_cols):
        """
        Remove clusters that are identical to their child.
        """
        import pandas as pd

        cluster_size = {}

        for cluster_col in cluster_cols:
            cluster_size.update(df.groupby(cluster_col).size())

        raw_path = set()

        for idx, row in df.iterrows():
            raw_path.add(':'.join(row[cluster_cols]))

        len(raw_path) # 473

        filtered_paths = []
        for rp_str in list(raw_path):
            rp = rp_str.split(':')
            fp = [rp[-1]]
            for i in range(len(rp) - 1, 0, -1):
                if cluster_size[rp[i-1]] > cluster_size[rp[i]]:
                    fp.append(rp[i-1])
            filtered_paths.append(list(reversed(fp)))
        
        return filtered_paths


    def get_nodes_and_edges_for_filtered_paths(filtered_paths):
        """
        Returns dataframes for nodes and edges of tree-structured graph representing hierarchical clustering 
        sliced at various levels.
        """
        import pandas as pd
        node_list = sorted(list({node for path in filtered_paths for node in path}))
        nodes_pdf = pd.DataFrame({'id':range(len(node_list)), 'cluster':node_list})
        node_id = {node:idx for node, idx in zip(node_list, range(len(node_list)))}

        parent_of = {}
        for fp in filtered_paths:
            for i in reversed(range(1, len(fp))):
                child = fp[i]
                parent = fp[i-1]
                parent_of[child] = parent
        edges_pdf = pd.DataFrame({'from':node_id[v], 'to':node_id[k], 'weight':1.0} for k,v in parent_of.items())

        return nodes_pdf, edges_pdf
        
        
    def featurize_sentences(all_sentences, sentence_transformer_model='all-mpnet-base-v2'):
        """
            Use SentenceTransformer locally.
            Vectors are converted to lists instead of numpy arrays because they are easier to save and restore to CSV format.
        """
        import pandas as pd
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/{sentence_transformer_model}')
        embedding_array = model.encode(all_sentences)
        return pd.Series([[x for x in v] for v in embedding_array])


    # Cluster info methods
    def get_cluster_centroids(df, vector_col, key_cols):
        """
        vector_col: the name of the column containing the embedding vectors.
        key_cols: list of columns containing cluster names.
        Example:
            cluster_cols = [f'cluster_{ltr}' for ltr in 'ABCDEF']
            thought_graph.get_cluster_centroids(document_sentence_vector, 'vector', cluster_cols)
        """
        # see https://stackoverflow.com/questions/47830763/python-dataframe-groupby-and-centroid-calculation
        import pandas as pd
        import numpy as np

        cluster_centroid_df_list = []

        for key_col in key_cols:
            centroids = df.groupby(key_col)[vector_col].apply(lambda x: np.mean(x.tolist(), axis=0))
            ccdf = pd.DataFrame({'cluster':[i for i in centroids.index], 'centroid': centroids.values})
            cluster_centroid_df_list.append(ccdf)

        return pd.concat(cluster_centroid_df_list).reset_index(drop=True)


    def get_candidate_names(cluster_text_pdf, cluster_col, text_col='sentence', 
                            sentence_sep=' ... ', max_ngram_length=3, top_n=1):
        """
        Use TF-IDF to find terms to use as candidate names for sections of text (especially clusters). 
        This function returns the cluster candidate names for each member of the cluster; use 'get_cluster_titles' 
        to summarize this by cluster.

        Example:
        cluster_sentence_pdf = article_sentence.copy()
        cluster_sentence_pdf['clusterA_name'] = get_candidate_names(cluster_sentence_pdf, cluster_col='clusterA', text_col='sentence')
        """
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        corpus = cluster_text_pdf[[cluster_col, text_col]]\
                        .groupby([cluster_col])[text_col]\
                        .transform(lambda s: sentence_sep.join(s))\
                        .values

        stop_words = 'english' # ['a', 'an', 'the', 'and', 'of', 'at', 'in', 'or', 'been']
        tfidf = TfidfVectorizer(ngram_range=(1, max_ngram_length), 
                                stop_words=stop_words, # sublinear_tf=True, # max_df=0.5,
                                lowercase=False)
        X = tfidf.fit_transform(corpus)
        feature_names = tfidf.get_feature_names()

        # give slight advantage to terms containing more words
        npX = np.array(X.todense())
        smidge = 1e-12  # just enough to break ties, not enough to affect real differences
        feature_tfidf_adjustment = [smidge * len(n.split(' ')) for n in feature_names]
        adjusted_X = np.array([npX[i,:]+feature_tfidf_adjustment for i in range(len(npX))])
        # candidate_names = [x for x in np.array(feature_names)[adjusted_X.argmax(axis=1)].tolist()]

        top_idx = [ [i for i in v[-top_n:][::-1]] for v in adjusted_X.argsort(axis=1)] # [-n:]
        candidate_names = [', '.join(x) for x in np.array(feature_names)[top_idx].tolist()]

        return candidate_names
    
   
    def get_cluster_key_terms(df, cluster_cols, cluster_title_cols):
        """
        Create a Pandas dataframe showing the candidate title for each cluster.

        Example:
            cluster_key_term_cols = [ccol + '_key_term' for ccol in cluster_cols]
            thought_graph.get_cluster_key_terms(dsv, cluster_cols, cluster_key_term_cols)
        """
        import pandas as pd
        cluster_title = {}

        for idx in range(len(cluster_cols)):
            ccol = cluster_cols[idx]
            title_col = cluster_title_cols[idx]
            for idx, row in df.iterrows():
                cluster_title[row[ccol]] = row[title_col]

        return pd.DataFrame([{'cluster':k, 'key_term': v} for k,v in cluster_title.items()])

    
    def get_cluster_text(df, cluster_cols, in_text_col, out_text_col):
        import pandas as pd
        cluster_text_df_list = []
        for ccol in cluster_cols:
            df[out_text_col] = df[[in_text_col, ccol]].groupby(ccol)[in_text_col].transform(lambda x: '\n'.join(x))
            cdf = df[[ccol, out_text_col]].drop_duplicates()
            cdf.columns = ['cluster', out_text_col]
            cluster_text_df_list.append(cdf)

        return pd.concat(cluster_text_df_list)


    def get_pairwise_edges_from_centroids(cc):
        import pandas as pd
        from scipy.cluster.hierarchy import ward, fcluster
        from scipy.spatial.distance import pdist, squareform, cosine
        from math import sqrt

        def bob_cosine_sim(u, v):
            return np.sum(u * v)/(sqrt(np.sum(u*u)) * sqrt(np.sum(v*v)))

        nodes = cc['cluster'].values
        n = len(nodes)
        m = int((n * (n-1))/2)
        X = cc['centroid'].tolist()
        pairwise_distance = pdist(X, metric='cosine') # these values are shifted so minimum distance is 0 (no negatives)
        pairwise_similarity = [ 1-d for d in pairwise_distance]
        # pairwise_similarity = pdist(X, metric=bob_cosine_sim) # takes about a minute
        from_list = [None] * m
        to_list = [None] * m
        p = 0
        for i in range(n - 1):
            for j in range(i+1, n):
                # print(f"i={i}, j={j}")
                from_list[p] = nodes[i]
                to_list[p] = nodes[j]
                p = p + 1
        return pd.DataFrame({'from':from_list, 'to':to_list, 'similarity':pairwise_similarity})


    def get_level_color_map(levels):
        def rainbow(N):
            import colorsys
            import math
            return ['#%02x%02x%02x' % tuple(math.floor(255 * j) for j in colorsys.hls_to_rgb(i/(1.5*N), 0.8, 1.0)) for i in range(N)]
        colors = rainbow(len(levels))
        return {levels[i]:colors[i] for i in range(len(levels))}
        