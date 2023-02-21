
class thought_graph:
    
    def get_document_sentences(text_files):
        import pandas as pd
        from spacy.lang.en import English
        import chardet
        nlp = English()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))

        doc_sent_dataframes_list = []

        for text_file in text_files:
            with open(text_file, 'rb') as in_txt:  # 'r', encoding='utf-8'
                content = in_txt.read()
                encoding = chardet.detect(content)['encoding']
                if encoding == 'utf-8':
                    content = str(content)
                else: # 'Windows-1252' is apparently the same as 'cp1251'
                    content = str(content.decode(encoding).encode('utf8'))  
            document_name = text_file.replace('_content.txt', '')
            sentences = [str(s) for s in nlp(content).sents] # sentences are 'spacy.tokens.span.Span' objects
            doc_sent_df = pd.DataFrame({'document': document_name, 'sentence': sentences})
            doc_sent_dataframes_list.append(doc_sent_df)

        return pd.concat(doc_sent_dataframes_list)

    
    def featurize_sentences(all_sentences):
        """
            Use SentenceTransformer locally.
            Vectors are converted to lists instead of numpy arrays because they are easier to save and restore to CSV format.
        """
        import pandas as pd
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embedding_array = model.encode(all_sentences)
        return pd.Series([[x for x in v] for v in embedding_array])
    
    
    def hclust_vectors(df, LETTERS='ABCDEF', max_threshold=4):
        from scipy.cluster.hierarchy import ward, fcluster
        from scipy.spatial.distance import pdist
        import math
        # cluster the sentence vectors at various levels
        X = df['vector'].tolist()
        y = pdist(X, metric='cosine')
        z = ward(y)

        # LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(len(LETTERS)):
            letter = LETTERS[i]
            col_name = f'cluster_{letter}'
            cluster_id = fcluster(z, max_threshold/2**i, criterion='distance')
            digits = 1 + math.floor(math.log10(max(cluster_id)))
            df[col_name] = [letter + str(cid).zfill(digits) for cid in cluster_id]

        cluster_cols = [c for c in df.columns if c.startswith('cluster_')]
        return df.sort_values(by=cluster_cols)
            
            
    def featurize_sentences_with_turing(all_sentences, my_secrets):
        """
        all_sentences: a Pandas Series of text items.
        my_secrets: dict containing 'ace_turing_primary_key'
        
        returns: Pandas dataframe with two columns:
                'sentence' lists the unique items from the input Series, and 
                'vector' gives the embedding for that item.
        
        example:
            sentence_vector = thought_graph.featurize_sentences(document_sentence['sentence'])
            # Merge with the original document_sentence dataframe:
            document_sentence['vector'] = document_sentence.merge(sentence_vector, how='left', on='sentence')
        """
        import http.client, urllib.request, urllib.parse, urllib.error, base64, json
        import pandas as pd
        from scipy.cluster.hierarchy import ward, fcluster
        from scipy.spatial.distance import pdist
        import math

        def send_turing_request(docs, conn, my_secrets):
            docs = [d.replace('"','\\"') for d in docs]
            body = '{{"queries": [{0}]}}'.format(', '.join([f'"{d}"' for d in docs]))
            headers = {'Content-Type': 'application/json', 'Ocp-Apim-Subscription-Key': my_secrets.ace_turing_primary_key,}
            params = urllib.parse.urlencode({})
            try:
                conn.request("POST", "/uni-genencoder/?%s" % params, body.encode('utf-8'), headers)
                response = conn.getresponse()
                result = response.read()
            except e:
                print(e)
                results = None
            return result

        unique_sentences = all_sentences.unique()
        
        all_results = []
        batch_size = 25

        conn = http.client.HTTPSConnection(my_secrets.ace_turing_endpoint) # 'turing-academic.azure-api.net'
        for i in range(0, len(unique_sentences), batch_size):
            doc_batch = unique_sentences[i:i + batch_size]
            print(f"item number {i}")
            batch_results = send_turing_request(doc_batch, conn, my_secrets)
            all_results.append(batch_results)

        conn.close()
        
        # process JSON results
        json_failure_count = 0
        results_df_list = []
        for i in range(len(all_results)):
            batch_results = all_results[i]
            rows = []
            try:
                br_obj = json.loads(batch_results)
                for b in br_obj:
                    rows.append({ 'sentence':b['query'], 'vector': b['vector'] })
                results_df_list.append(pd.DataFrame(rows))
            except:
                print("Failed to load JSON for sentence: " + unique_sentences[i])
                print(str(batch_results))
                print()
                json_failure_count = json_failure_count + 1

        if json_failure_count > 0:
            print(f"{json_failure_count} results failed to load from JSON")
        sentence_vector = pd.concat(results_df_list)

        # cluster the sentence vectors at various levels
        X = sentence_vector['vector'].tolist()
        y = pdist(X, metric='cosine')
        z = ward(y)

        max_threshold = 4
        LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(6):
            letter = LETTERS[i]
            col_name = f'cluster_{letter}'
            cluster_id = fcluster(z, max_threshold/2**i, criterion='distance')
            digits = 1 + math.floor(math.log10(max(cluster_id)))
            sentence_vector[col_name] = [letter + str(cid).zfill(digits) for cid in cluster_id]

        return sentence_vector

    
    def count_clusters_at_all_levels(dsc):
        cluster_cols = [c for c in dsc.columns if c.startswith("cluster_")]
        for col in cluster_cols:
            num_clusters = len(set(dsc[col]))
            print(f"{col}: {num_clusters}")
    
    
    # tree graph functions
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
    
    
    def export_edge_thresholds_graph_to_vis_js(nodes_df, edges_df, title, html_file_name):
        """
        Generate vis_js graph from cluster data using between-centroid distance as edge weights, and write to HTML file.

        Example:
          nodes_df['group'] = [1 if l.startswith('lab_') else 2 if l.startswith('d') else 0 for l in nodes_df['label']]
          export_edge_thresholds_graph_to_vis_js(nodes_df, edges_df, 'Cluster Graph', 'cluster_graph.html')
        """
        max_weight = 1.0 # np.quantile(edges_df['weight'], 0.95)

        nodes_str = nodes_df.to_json(orient='records')
        edges_str = edges_df.to_json(orient='records')

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
        
        
    def collapsible_clusters(nodes_df, edges_df, clusters_df, title, html_file_name):
        """
        Generate interactive graph with one level of collapsible clusters.
        
        nodes_df: label, title, shape, cid (but NOT color)
        edges_df: from, to, strength (I think - maybe weight?)
        clusters_df: cid, color (must be unique for each cluster), label, title
        Note that colors must be assigned to the clusters in the 'clusters_df' dataframe; these color names are what the nodes will be grouped by.
        The consituent nodes will be colored by the clusters they belong to.
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

            
    
    def extract_pdf_content_to_text_files(pdf_directory, text_directory, my_secrets):
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.formrecognizer import DocumentAnalysisClient
        from collections import Counter
        import os
        import re
        import pandas as pd
        
        def analyze_document(document_path, document_analysis_client):
            with open(document_path, "rb") as f:
                poller = document_analysis_client.begin_analyze_document(
                    "prebuilt-document", document=f
                )
            result = poller.result()
            return result


        def get_entities_table(result):
            return pd.DataFrame([entity.to_dict() for entity in result.entities])


        def preprocess_content(txt):
            # There are 2 main reasons a line will have only a few words: it is a section heading, or it is the end of a sentence.

            lines = txt.split('\n')
            line_counter = Counter(lines)
            lines = [line for line in lines if line_counter[line] <= 4]
            lines = [re.sub('-$', '', line) for line in lines]
            lines = [l + '.' if (len(l) > 5 and len(l.split(' ')) < 4 and not l.endswith('.')) else l for l in lines] # end apparent section headings with a period
            # Assume a line is a section heading if:
            # * it has more than 5 characters
            # * it has fewer than 4 words
            # * it does not already end in a period
            reconstituted = ' '.join(lines)
            reconstituted = reconstituted.replace('- ', '')
            return reconstituted

        document_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

        da_client = DocumentAnalysisClient(
            endpoint = my_secrets.form_recognizer_endpoint, credential=AzureKeyCredential(my_secrets.form_recognizer_key)
        )
        
        os.makedirs(text_directory, exist_ok=True)
        for document_file in document_files:
            print(f"Extracting text content from '{document_file}'")
            document_path = pdf_directory + document_file
            outfile_base = document_file.replace('.pdf', '').replace('  ', ' ').replace(' ', '_')
            content_text_file = text_directory + outfile_base + '_content.txt'
            result = analyze_document(document_path, da_client)
            # entities_file = outfile_base + '_entities.csv'
            # entities_table = get_entities_table(result)
            # entities_table.to_csv(entities_file)
            pp_content = preprocess_content(result.content)
            print(f"writing content to '{content_text_file}'")
            with open(content_text_file, 'w', encoding='utf-8') as txt_fh:
                txt_fh.write(pp_content)

        da_client.close()
        

    def get_column_types(df):
        """
        Is something like this built into Pandas?
        """
        import pandas as pd
        return pd.DataFrame([{'column': col, 'type':type(df[col][0]).__name__} for col in df.columns])


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
        

    def split_into_sentences(text):
        """
        To Do: remove reference numbers, like '[1,2,3-5]'
        New patterns: 'et al.', floating point numbers
        https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
        """
        import re
        abbreviations = "(et al|etc)[.]"
        alphabets= "([A-Za-z])"
        prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        websites = "[.](com|net|org|io|gov)"

        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub("(\\d+)[.](\\d+)","\\1<prd>\\2",text)
        text = re.sub(abbreviations,"\\1<prd>",text)
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences
