"""Exact collision bridge and inductive atlas verification."""
if not __debug__:
    raise RuntimeError("Optimized Python disables checks; run without -O or -OO")
import json,itertools as it,math,time,hashlib
from pathlib import Path
from fractions import Fraction as Q
import numpy as np
import networkx as nx
P=Path(__file__).resolve().parent;O=Path(__file__).resolve().parent
def pairs(n):return list(it.combinations(range(n),2))
def code(A,p):return sum(int(A[p[i],p[j]])<<b for b,(i,j) in enumerate(pairs(len(p))))
def edges(c,n):return [(i,j) for b,(i,j) in enumerate(pairs(n)) if c>>b&1]
def enc(E,n):
    E={tuple(sorted(e)) for e in E};return sum(1<<b for b,e in enumerate(pairs(n)) if e in E)
def tf(G):return not sum(nx.triangles(G).values())
def invariant(G):return tuple(sorted(dict(G.degree()).values()))
# Build complete TF catalogs inductively, without using graph_atlas_g.
catalog=[nx.empty_graph(0)];numbers=[1]
for n in range(1,9):
    nextcat=[];buckets={}
    for G in catalog:
        for mask in range(1<<(n-1)):
            N=[v for v in range(n-1) if mask>>v&1]
            if any(G.has_edge(a,b) for a,b in it.combinations(N,2)):continue
            H=G.copy();H.add_node(n-1);H.add_edges_from((v,n-1) for v in N)
            b=buckets.setdefault(invariant(H),[])
            if not any(nx.is_isomorphic(H,J) for J in b):b.append(H);nextcat.append(H)
    catalog=nextcat;numbers.append(len(catalog));print('inductive triangle-free catalog',n,len(catalog),flush=True)
S=np.load(P/'trianglefree-eight-graphs.npy');sg=[nx.from_numpy_array(H) for H in S];buckets={}
for i,G in enumerate(sg):buckets.setdefault(invariant(G),[]).append(i)
assert numbers==[1,1,2,3,7,14,38,107,410]
assert all(sum(nx.is_isomorphic(G,sg[i]) for i in buckets[invariant(G)])==1 for G in catalog)
A=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.int64);weights=[1,2,4];w=[Q(x,7) for x in weights];counts=[0]*410;cache={}
for p in it.product(range(3),repeat=8):
    c=code(A,p)
    if c not in cache:
        G=nx.Graph();G.add_nodes_from(range(8));G.add_edges_from(edges(c,8))
        match=[i for i in buckets[invariant(G)] if nx.is_isomorphic(G,sg[i])];assert len(match)==1;cache[c]=match[0]
    counts[cache[c]]+=math.prod(weights[i] for i in p)
assert sum(counts)==7**8
prob=[Q(c,7**8) for c in counts]
meta=json.loads((P/'filtered-eight-data.json').read_text());raw=np.load(P/'filtered-eight-grams.npz')
checks=0;colliding_nonzero=0
for group in meta['groups']:
    k,l=group['k'],group['l'];m=l-k;d=group['dimension'];index={c:i for i,c in enumerate(group['flag_codes'])}
    # Canonicalization keeps root order and permutes only the free labels.
    def flag(p):
        E=edges(code(A,p),l)
        c=min(enc([(r[a],r[b]) for a,b in E],l) for tail in it.permutations(range(k,l)) for r in [list(range(k))+list(tail)])
        return index[c]
    direct={}
    for root in it.product(range(3),repeat=k):
        if code(A,root)!=group['type_code']:continue
        rp=math.prod(w[i] for i in root);v={}
        for free in it.product(range(3),repeat=m):
            i=flag(root+free);v[i]=v.get(i,Q(0))+math.prod(w[j] for j in free)
        if len(set(root))<len(root) and v:colliding_nonzero+=1
        for i,a in v.items():
            for j,b in v.items():direct[i,j]=direct.get((i,j),Q(0))+rp*a*b
    source={}
    for hi,p in enumerate(prob):
        if not p:continue
        C=raw[group['key']][hi]
        for i,j in zip(*np.nonzero(C)):
            ij=(int(i),int(j));source[ij]=source.get(ij,Q(0))+p*Q(int(C[i,j]),40320)
    assert source==direct,group['key'];checks+=d*d
print('collision source/conditional flag equality: all',checks,'entries; repeated root cases',colliding_nonzero,flush=True)
f=json.loads((P/'filtered-eight-degree-family.json').read_text());n=3;degree=[sum(int(A[z,t])*w[t] for t in range(n)) for z in range(n)]
def T(v):return [sum(int(A[z,t])*w[t]*v[t] for t in range(n)) for z in range(n)]
F=[[Q(0) for j in range(5)] for i in range(5)];D=[[Q(0) for j in range(5)] for i in range(5)]
for x,y in it.product(range(n),repeat=2):
    if not A[x,y]:continue
    ff=[Q(int(A[z,x]-A[z,y])) for z in range(n)];hh=[Q(int(A[z,x]+A[z,y])) for z in range(n)];tt=T(ff);tt2=T(tt);de=degree[x]-degree[y];g=[degree[z]-(degree[x]+degree[y])/2 for z in range(n)]
    U=[[tt[z]-ff[z]/50 for z in range(n)],[de*a for a in hh],[g[z]*ff[z] for z in range(n)],[tt2[z]+Q(3,50)*tt[z]-ff[z]/625 for z in range(n)]];U.append([g[z]*U[0][z]+de*de*ff[z]/4 for z in range(n)])
    for i,j in it.product(range(5),repeat=2):
        norm=sum(w[z]*U[i][z]*U[j][z] for z in range(n));tv=T(U[j]);en=sum(w[z]*U[i][z]*(degree[z]*U[j][z]+tv[z]) for z in range(n))
        D[i][j]+=w[x]*w[y]*norm;F[i][j]+=w[x]*w[y]*(en-Q(7,50)*norm)
for field,direct in [('objective_matrices',F),('norm_matrices',D)]:
    for i,j in it.product(range(5),repeat=2):assert sum(prob[h]*Q(f[field][h][i][j]) for h in range(410))==direct[i][j]
result={'inductive_catalog_numbers':numbers,'independently_generated_eight_classes':410,'weighted_base':'P3 with masses 1/7,2/7,4/7','all_maps':3**8,'source_classes_positive':sum(x>0 for x in counts),'conditional_flag_entries':checks,'colliding_root_events':colliding_nonzero,'spectral_source_vs_direct_exact_entries':50,'status':'passed'}
(O/'collision-source-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
