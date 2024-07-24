import graphviz

dot_code = '''
digraph Tree {
node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;
edge [fontname=helvetica] ;
0 [label=<node &#35;0<br/>h &le; 9.0<br/>samples = 100.0%<br/>prediction_value = 0.0>, fillcolor="#ea9b61"] ;
1 [label=<&#35;1<br/>h &le; 2.84<br/>71.8%<br/>0.04>, fillcolor="#e99456"] ;
0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
2 [label=<&#35;2<br/>h &le; 2.62<br/>11.8%<br/>-0.17>, fillcolor="#f0b991"] ;
1 -> 2 ;
3 [label=<&#35;3<br/>4.8%<br/>0.01>, fillcolor="#ea995e"] ;
2 -> 3 ;
4 [label=<&#35;4<br/>7.0%<br/>-0.29>, fillcolor="#f5ceb3"] ;
2 -> 4 ;
5 [label=<&#35;5<br/>C &le; 15.0<br/>60.0%<br/>0.08>, fillcolor="#e78c4b"] ;
1 -> 5 ;
6 [label=<&#35;6<br/>41.2%<br/>0.05>, fillcolor="#e89253"] ;
5 -> 6 ;
7 [label=<&#35;7<br/>18.8%<br/>0.15>, fillcolor="#e58139"] ;
5 -> 7 ;
8 [label=<&#35;8<br/>h &le; 10.11<br/>28.2%<br/>-0.1>, fillcolor="#eeac7d"] ;
0 -> 8 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
9 [label=<&#35;9<br/>C &le; 7.0<br/>3.2%<br/>-0.54>, fillcolor="#fefaf6"] ;
8 -> 9 ;
10 [label=<&#35;10<br/>2.2%<br/>-0.57>, fillcolor="#ffffff"] ;
9 -> 10 ;
11 [label=<&#35;11<br/>1.0%<br/>-0.47>, fillcolor="#fbede3"] ;
9 -> 11 ;
12 [label=<&#35;12<br/>h &le; 10.39<br/>25.0%<br/>-0.04>, fillcolor="#eca26e"] ;
8 -> 12 ;
13 [label=<&#35;13<br/>21.2%<br/>-0.0>, fillcolor="#ea9b62"] ;
12 -> 13 ;
14 [label=<&#35;14<br/>3.8%<br/>-0.29>, fillcolor="#f5ceb1"] ;
12 -> 14 ;
}
'''

# Create a graph from the dot code
graph = graphviz.Source(dot_code)

# Save the graph to a file and render it
graph.render('tree_graph', format='png', cleanup=True)
