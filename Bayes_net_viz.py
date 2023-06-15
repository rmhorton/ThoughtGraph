
import pandas as pd
import numpy as np

def get_nodes_and_edges_from_modelstring(modelstring):
    node_list = ['zero']
    edge_list = []

    for np_str in modelstring[1:-1].split(']['):
        if '|' in np_str:
            node, parent_str = np_str.split('|')
            parents = parent_str.split(':')
        else:
            node, parents = np_str, None
        node_list.append(node)
        node_id = node_list.index(node)
        if parents is not None:
            for parent in parents:
                # edge_list.append({'from': parent, 'to': node})
                parent_id = node_list.index(parent)
                edge_list.append({'from': parent_id, 'to': node_id})

    nodes = pd.DataFrame({'id': range(len(node_list)), 'label': node_list})[1:]
    edges = pd.DataFrame(edge_list)
    edges['weight'] = 1
    return nodes, edges


def get_vis_js_html(nodes_df, edges_df):
    """
    Generate HTML encoding vis_js graph from Pandas dataframes of nodes and edges.
    """
    nodes_str = nodes_df.to_json(orient='records')
    edges_str = edges_df.to_json(orient='records')
    
    max_weight = max(edges_df['weight'])

    html_string = ( 
    f'    <script type="text/javascript">NODE_LIST={nodes_str};FULL_EDGE_LIST={edges_str};</script>\n'
    '\n'
        '\n'
    '        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>\n'
    '        <script type="text/javascript">\n'
    '            function hello(){console.info("hello")}\n'
    '\n'
    '            const sign_color = {pos:"blue", neg:"red", zero:"black"}\n'
    '            const options = {physics:{maxVelocity: 1, minVelocity: 0.01}}\n'
    '            var edgeFilterSlider\n'
    '            var mynetwork\n'
    '            var motion_flag = false\n'
    '            function toggle_motion(){\n'
    '                motion_flag = !motion_flag\n'
    '                mynetwork.setOptions( { physics: motion_flag } )\n'
    '            }\n'
    '\n'
    '            function edgesFilter(edge){\n'
    '                return edge.value >= edgeFilterSlider.value\n'
    '            }\n'
    '\n'
    '            function init_network(){\n'
    '                document.getElementById("min_edge_weight_display").value = 0.5\n'
    '                document.getElementById("min_edge_weight").onchange = function(){\n'
    '                    document.getElementById("min_edge_weight_display").value = this.value\n'
    '                }\n'
    '\n'
    '                edgeFilterSlider = document.getElementById("min_edge_weight")\n'
    '                edgeFilterSlider.addEventListener("change", (e) => {edgesView.refresh()})\n'
    '                var container = document.getElementById("mynetwork")\n'
    '                var EDGE_LIST = []\n'
    '                for (var i = 0; i < FULL_EDGE_LIST.length; i++) {\n'
    '                    var edge = FULL_EDGE_LIST[i]\n'
    '                    edge["value"] = Math.abs(edge["weight"])\n'
    '                    edge["title"] = "weight " + edge["weight"]\n'
    '                    edge["sign"] = (edge["weight"] < 0) ? "neg" : "pos";\n'
    '                    edge["color"] = {color: sign_color[edge["sign"]] };\n'
    '                    edge["arrows"] = "to"\n'
    '                    EDGE_LIST.push(edge)\n'
    '                }\n'
    '\n'
    '                var nodes = new vis.DataSet(NODE_LIST)\n'
    '                var edges = new vis.DataSet(EDGE_LIST)\n'
    '                var nodesView = new vis.DataView(nodes)\n'
    '                var edgesView = new vis.DataView(edges, { filter: edgesFilter })\n'
    '                var data = { nodes: nodesView, edges: edgesView }\n'
    '                mynetwork = new vis.Network(container, data, options)\n'
    '\n'
    '            }\n'
    '            init_network()\n'
    '        </script>\n'
    '        <style type="text/css">#mynetwork {width: 100%; height: 500px; border: 3px}</style>\n'
    '        <button onclick=toggle_motion()>Toggle motion</button>\n'
    '        <div class="slidercontainer">\n'
    '            <label>minimum edge weight:\n'
    f'                <input type="range" min="0" max="{max_weight}" value="{max_weight/2}" step="{max_weight/100}" class="slider" id="min_edge_weight">\n'
    '                <input type="text" id="min_edge_weight_display" size="2">\n'
    '            </label>\n'
    '        </div>\n'
    '        <div id="mynetwork"></div>\n'

    )
    return html_string
        
    
def export_to_vis_js(nodes_df, edges_df, title, html_file_name):
    """
    Generate vis_js graph from Pandas dataframes of nodes and edges, and write to HTML file.
    """
    
    vis_js_html = get_vis_js_html(nodes_df, edges_df)
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


def direction_to_color(d):
    # d is a value between 0 and 1; turn it into a color
    import colorsys
    rgb = colorsys.hsv_to_rgb(d/2, 0.75, 0.50)
    return '#' + "".join("%02X" % round(i*255) for i in rgb)


def reformat_edge_table(edge_table):
    # The input edge table identifies nodes by name; reformat to have separate node and edge tables.
    # Renames 'strength' to 'weight', multiplies by 'direction' if present
    next_node_id = 1
    node_id = {}
    new_rows = []
    for idx, row in edge_table.iterrows():
        if row['from'] not in node_id:
            node_id[ row['from'] ] = next_node_id
            next_node_id += 1
        if row['to'] not in node_id:
            node_id[ row['to'] ] = next_node_id
            next_node_id += 1
        from_id = node_id[ row['from'] ]
        to_id = node_id[ row['to'] ]
        row['from'] = from_id
        row['to' ] = to_id
        if 'direction' in row: # bootstrap edges
            row['weight'] = row['strength'] * row['direction']
        else: # arc.strength
            row['weight'] = row['strength']
        new_rows.append(row)
    new_edges = pd.DataFrame(new_rows)  # .rename(columns={'strength':'weight'})
    # new_edges['color'] = [direction_to_color(d) for d in new_edges['direction']]
    new_nodes = pd.DataFrame({'id': node_id.values(), 'label':node_id.keys()})
    return new_nodes, new_edges[['from', 'to', 'weight']] # 'color'


def decorate_nodes(nodes):
    nodes['color'] = ['#9090F0' if lbl.endswith('_flag') else '#F09090' for lbl in nodes['label']]
    nodes['shape'] = ['ellipse' if lbl.endswith('_flag') else 'box' for lbl in nodes['label']]
    return nodes
