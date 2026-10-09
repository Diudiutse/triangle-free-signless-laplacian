"""Portable exact verifier adapted from the third workspace review.
Rebuilds symbolic expansions, source lifts, flag catalogs/counts, and the certificate.
Run with ordinary Python 3, NumPy and NetworkX.
"""
if not __debug__:
    raise RuntimeError("Optimized Python disables checks; run without -O or -OO")
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,hashlib,time,itertools as it,math
from pathlib import Path
from fractions import Fraction as R
import numpy as np
import networkx as nx
sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parent
OUT=Path(__file__).resolve().parent
start=time.time(); results={}
def log(*x): print(*x,flush=True)
def load(n): return json.loads((ROOT/n).read_text(encoding='utf-8'))
def pairs(n): return list(it.combinations(range(n),2))
def decode(c,n): return [(a,b) for i,(a,b) in enumerate(pairs(n)) if c>>i&1]
def encode(E,n):
    E={tuple(sorted(e)) for e in E}
    return sum(1<<i for i,e in enumerate(pairs(n)) if (a:=e) in E)
def graph(c,n):
    g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from(decode(c,n));return g
def tf(E):
    E={tuple(sorted(e)) for e in E}; V=set(sum(([a,b] for a,b in E),[]))
    return not any({(a,b),(a,c),(b,c)}<=E for a,b,c in it.combinations(sorted(V),3))
manifest=load('manifest.json')['sha256']
actual={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in manifest}
assert actual==manifest;results['input_sha256']=actual;log('All distributed input SHA256 values match')
cert=load('filtered-sharp-exact-certificate.json'); fam=load('filtered-eight-degree-family.json');meta=load('filtered-eight-data.json')
sources=np.load(ROOT/'trianglefree-eight-graphs.npy'); raw=np.load(ROOT/'filtered-eight-grams.npz')
assert sources.shape==(410,8,8)
for H in sources:
    assert np.array_equal(H,H.T) and np.all(np.diag(H)==0) and set(np.unique(H))<={0,1}
    assert np.trace(H@H@H)==0
SG=[nx.from_numpy_array(H) for H in sources]
def inv(g): return tuple(sorted(dict(g.degree()).values()))
buckets={}
for i,g in enumerate(SG):
    b=buckets.setdefault(inv(g),[])
    assert not any(nx.is_isomorphic(g,SG[j]) for j in b);b.append(i)
# Delete one vertex from any eight-graph: its seven-graph occurs in the atlas.
# Conversely extend every atlas seven-graph by every independent neighborhood.
coverage=set();extensions=0
for g in nx.graph_atlas_g():
    if len(g)!=7 or sum(nx.triangles(g).values()):continue
    for mask in range(128):
        N=[i for i in range(7) if mask>>i&1]
        if any(g.has_edge(a,b) for a,b in it.combinations(N,2)):continue
        h=g.copy();h.add_node(7);h.add_edges_from((v,7) for v in N);extensions+=1
        matches=[j for j in buckets.get(inv(h),[]) if nx.is_isomorphic(h,SG[j])]
        assert len(matches)==1;coverage.add(matches[0])
assert coverage==set(range(410));results['source_catalog']={'sources':410,'extensions':extensions,'complete_via_seven_atlas':True};log('source catalog complete:',extensions,'extensions')
# Derive symbolic vectors from arithmetic expressions; all internal factors get
# their own variable names. A monomial is an edge list, not a project edge code.
counter=it.count(10)
def add(*polys): return sum((p for p in polys),[])
def scale(c,p): return [(c*a,e) for a,e in p]
def mul(p,q): return [(a*b,e+f) for a,e in p for b,f in q]
def edge(a,b): return [(R(1),[(a,b)])]
def degree(z): return edge(z,next(counter))
def f(z): return add(edge(z,'x'),scale(-1,edge(z,'y')))
def h(z): return add(edge(z,'x'),edge(z,'y'))
def delta(): return add(degree('x'),scale(-1,degree('y')))
def g(z): return add(degree(z),scale(R(-1,2),degree('x')),scale(R(-1,2),degree('y')))
def T(pfun,z):
    a=next(counter);return mul(edge(z,a),pfun(a))
def U(i,z):
    if i==0:return add(T(f,z),scale(R(-1,50),f(z)))
    if i==1:return mul(delta(),h(z))
    if i==2:return mul(g(z),f(z))
    if i==3:return add(T(lambda s:T(f,s),z),scale(R(3,50),T(f,z)),scale(R(-1,625),f(z)))
    return add(mul(g(z),U(0,z)),scale(R(1,4),mul(mul(delta(),delta()),f(z))))
storedkeys=sorted({(t['vertices'],t['edge_code']) for b in ['hom_terms','norm_hom_terms'] for row in fam[b] for p in row for t in p})
representatives={k:graph(k[1],k[0]) for k in storedkeys}; byinv={}
for k,G in representatives.items(): byinv.setdefault((len(G),inv(G)),[]).append(k)
iso_cache={}
def key(E):
    E=set(tuple(sorted(e,key=str)) for e in E)
    vs=sorted(set(it.chain.from_iterable(E)),key=str); m={v:i for i,v in enumerate(vs)}
    ee={tuple(sorted((m[a],m[b]))) for a,b in E}
    if not tf(ee):return None
    code=encode(ee,len(vs)); kk=(len(vs),code)
    if kk not in iso_cache:
        G=graph(code,len(vs));matches=[k for k in byinv.get((len(vs),inv(G)),[]) if nx.is_isomorphic(G,representatives[k])]
        assert len(matches)<=1,('duplicate stored graph classes',kk,matches)
        if not matches:
            representatives[kk]=G;byinv.setdefault((len(vs),inv(G)),[]).append(kk);matches=[kk]
        iso_cache[kk]=matches[0]
    return iso_cache[kk]
def expand(i,j,norm=False):
    terms=mul(edge('x','y'),mul(U(i,'z'),U(j,'z')))
    if not norm:
        terms=add(mul(edge('x','y'),mul(degree('z'),mul(U(i,'z'),U(j,'z')))),mul(mul(edge('x','y'),edge('z','t')),mul(U(i,'z'),U(j,'t'))),scale(R(-7,50),terms))
    out={}
    for a,e in terms:
        k=key(e)
        if k is not None:out[k]=out.get(k,R(0))+a
    return {k:a for k,a in out.items() if a}
P={}
for b,norm in [('hom_terms',False),('norm_hom_terms',True)]:
    P[b]=[[expand(i,j,norm) for j in range(5)] for i in range(5)]
    for i in range(5):
        for j in range(5):
            saved={(t['vertices'],t['edge_code']):R(t['coefficient']) for t in fam[b][i][j]}
            assert P[b][i][j]==saved,(b,i,j)
results['symbolic_expansion']={'entries':50,'hom_graphs':len(storedkeys),'maximum_variables':max(k[0] for k in storedkeys)};log('independent symbolic expansion: all 50 polynomials exact')
# Independently lift each monomial into the eight-vertex induced source basis.
perms={n:np.array(list(it.permutations(range(8),n)),dtype=np.int16) for n in {k[0] for k in storedkeys}}
lift={}
for idx,(n,c) in enumerate(storedkeys):
    pp=perms[n]; vals=[]
    for H in sources:
        ok=np.ones(len(pp),dtype=bool)
        for a,b in decode(c,n):ok &= H[pp[:,a],pp[:,b]]==1
        vals.append(R(int(ok.sum()),len(pp)))
    lift[n,c]=vals
    if idx%15==0:log('source lifts',idx+1,'/',len(storedkeys))
for b,field in [('hom_terms','objective_matrices'),('norm_hom_terms','norm_matrices')]:
    for t in range(410):
        for i in range(5):
            for j in range(5):assert sum(a*lift[k][t] for k,a in P[b][i][j].items())==R(fam[field][t][i][j])
results['source_lift_entries']=20500;log('all 20500 source spectral entries exact')
# Enumerate flag catalogs with roots FIXED, independently relabel free vertices.
groups=meta['groups']; pp=perms[8]; gramchecks=0
for k in range(7):
    gl=[q for q in groups if q['k']==k];l=(8+k)//2;m=l-k
    expected_types=set()
    for G in nx.graph_atlas_g():
        if len(G)==k and not sum(nx.triangles(G).values()):
            expected_types.add(min(encode([(p[a],p[b]) for a,b in G.edges()],k) for p in it.permutations(range(k))))
    assert {q['type_code'] for q in gl}==expected_types
    size=1<<len(pairs(l)); gid=np.full(size,-1,dtype=np.int16); fid=np.full(size,-1,dtype=np.int16);offset=[];total=0
    for z,q in enumerate(gl):
        assert q['l']==l
        root=decode(q['type_code'],k); extra=[e for e in pairs(l) if e[1]>=k]; maps={}
        for mask in range(1<<len(extra)):
            E=root+[e for j,e in enumerate(extra) if mask>>j&1]
            if not tf(E):continue
            c=encode(E,l)
            canonical=min(encode([(p[a],p[b]) for a,b in E],l) for tail in it.permutations(range(k,l)) for p in [list(range(k))+list(tail)])
            maps[c]=canonical
        flags=sorted(set(maps.values()));assert flags==q['flag_codes'] and len(flags)==q['dimension']
        inds={c:i for i,c in enumerate(flags)}
        for c,fc in maps.items():gid[c]=z;fid[c]=inds[fc]
        offset.append(total);total+=len(flags)**2
    offsets=np.array(offset); dims=np.array([q['dimension'] for q in gl]);pos1=list(range(l));pos2=list(range(k))+list(range(l,l+m))
    for hi,H in enumerate(sources):
        def codes(pos):
            out=np.zeros(40320,dtype=np.int64)
            for bit,(a,b) in enumerate(pairs(l)):out+=H[pp[:,pos[a]],pp[:,pos[b]]].astype(np.int64)*(1<<bit)
            return out
        c1,c2=codes(pos1),codes(pos2); a,b=gid[c1],gid[c2]; valid=(a>=0)&(a==b)
        v=a[valid]; index=offsets[v]+fid[c1[valid]]*dims[v]+fid[c2[valid]]
        counts=np.bincount(index,minlength=total)
        for z,q in enumerate(gl):
            d=q['dimension'];got=counts[offset[z]:offset[z]+d*d].reshape(d,d)
            assert np.array_equal(got,raw[q['key']][hi]),('flag mismatch',k,q['key'],hi);gramchecks+=1
    log('flag roots',k,'all catalogs and',410*len(gl),'source matrices exact')
assert gramchecks==27060;results['flag_source_matrices']=gramchecks
# Exact certificate contraction, computed independently from explicit middle
# matrices and their congruences. Never consume saved corrected source totals.
q=int(cert['correction_denominator']);B=int(cert['global_source_denominator']);assert q>0 and B>0
corrections={};seen=set()
for c in cert['corrections']:
    tag=c['key'];i,j=c['i'],c['j'];assert i<=j and (tag,i,j) not in seen;seen.add((tag,i,j))
    corrections.setdefault(tag,{})[i,j]=int(c['numerator'])
assert set(corrections)<={g['key'] for g in groups}|{'spectral'}
ddrows=0;minmargin=q
def physical(tag,V):
    global ddrows,minmargin
    d=V.shape[1]; C=corrections.get(tag,{})
    M={}
    for (i,j),v in C.items():
        assert 0<=i<=j<d;M[i,j]=v;M[j,i]=v
    for i in range(d):
        margin=q+M.get((i,i),0)-sum(abs(M.get((i,j),0)) for j in range(d) if i!=j)
        assert margin>0;minmargin=min(minmargin,margin);ddrows+=1
    S=np.asarray(V@V.T,dtype=object)*q
    for (i,j),v in C.items():
        for a in range(len(V)):
            for b in range(len(V)):
                S[a,b]+=v*(V[a,i]*V[b,j]+(V[a,j]*V[b,i] if i!=j else 0))
    assert np.array_equal(S,S.T);return S
totals=[0]*410;cg={x['key']:x for x in cert['groups']};assert len(cg)==66 and set(cg)=={g['key'] for g in groups}
for gi,g0 in enumerate(groups):
    g=cg[g0['key']];V=np.array([[int(x) for x in row] for row in g['V_integer']],dtype=object);assert V.shape[0]==g0['dimension'];a=int(g['denominator']);assert a>0
    S=physical(g['key'],V);assert B%(a*a*40320)==0;factor=B//(a*a*40320)
    for hi,C in enumerate(raw[g['key']]):
        totals[hi]+=factor*sum(int(C[i,j])*int(S[i,j]) for i,j in zip(*np.nonzero(C)))
    if gi%10==0:log('certificate congruences',gi+1,'/66')
L=np.array([[int(x) for x in row] for row in cert['spectral_L_integer']],dtype=object);b=int(cert['spectral_denominator']);assert L.shape==(5,5) and b>0
assert all(L[i,j]==0 for i in range(5) for j in range(i+1,5)) and all(L[i,i]>0 for i in range(5))
Wq=physical('spectral',L);assert Wq[0,0]==q*b*b
for hi,F in enumerate(fam['objective_matrices']):
    z=sum(R(F[i][j])*int(Wq[i,j]) for i in range(5) for j in range(5))*R(B,b*b)
    assert z.denominator==1;totals[hi]+=z.numerator
assert totals==[0]*410;assert str(minmargin)==cert['strict_diagonal_dominance_minimum_numerator']
results['certificate']={'zero_source_numerators':410,'strict_dd_rows':ddrows,'spectral_positive_definite':True,'minimum_margin':str(R(minmargin,q))};log('certificate all 410 exact zeros; SDD rows',ddrows)
# Collision-inclusive exact cross-layer tests. Weighted three-class path, isolated
# class, and odd cycle: sum over all maps, never injective maps of base classes.
def hom_weighted(A,w,n,c):
    E=decode(c,n);out=R(0)
    for p in it.product(range(len(w)),repeat=n):
        if all(A[p[a],p[b]] for a,b in E):out+=math.prod(w[x] for x in p)
    return out
cases=[('path3-weighted',nx.path_graph(3),[R(1,7),R(2,7),R(4,7)]),('edge-isolate',nx.empty_graph(3),[R(1,4),R(1,4),R(1,2)]),('cycle5-uniform',nx.cycle_graph(5),[R(1,5)]*5)]
cases[1][1].add_edge(0,1);collision_results=[]
for name,G,w in cases:
    A=nx.to_numpy_array(G,dtype=np.int16);n=len(w);d=[sum(R(int(A[z,t]))*w[t] for t in range(n)) for z in range(n)]
    def Tvec(v):return [sum(R(int(A[z,t]))*w[t]*v[t] for t in range(n)) for z in range(n)]
    D=[[R(0) for j in range(5)] for i in range(5)];F=[[R(0) for j in range(5)] for i in range(5)]
    for x,y in it.product(range(n),repeat=2):
        if not A[x,y]:continue
        ff=[R(int(A[z,x]-A[z,y])) for z in range(n)];hh=[R(int(A[z,x]+A[z,y])) for z in range(n)];tt=Tvec(ff);tt2=Tvec(tt);de=d[x]-d[y];gg=[d[z]-(d[x]+d[y])/2 for z in range(n)]
        us=[[tt[z]-ff[z]/50 for z in range(n)],[de*h0 for h0 in hh],[gg[z]*ff[z] for z in range(n)],[tt2[z]+R(3,50)*tt[z]-ff[z]/625 for z in range(n)]]
        us.append([gg[z]*us[0][z]+de*de*ff[z]/4 for z in range(n)])
        assert us[0][x]-us[0][y]==d[x]+d[y]+R(1,25)
        for i,j in it.product(range(5),repeat=2):
            norm=sum(w[z]*us[i][z]*us[j][z] for z in range(n));tuj=Tvec(us[j]);energy=sum(w[z]*us[i][z]*(d[z]*us[j][z]+tuj[z]) for z in range(n))
            D[i][j]+=w[x]*w[y]*norm;F[i][j]+=w[x]*w[y]*(energy-R(7,50)*norm)
    values={k:hom_weighted(A,w,*k) for k in storedkeys}
    for block,actual in [('hom_terms',F),('norm_hom_terms',D)]:
        for i,j in it.product(range(5),repeat=2):assert sum(a*values[k] for k,a in P[block][i][j].items())==actual[i][j],(name,block,i,j)
    collision_results.append({'case':name,'exact_entries':50,'nonzero_U0_endpoint':True});log('all-map rational collision case',name,'50 exact entries')
results['collision_tests']=collision_results
HS=np.load(ROOT/'higman_sims_adjacency.npy').astype(np.int64);assert HS.shape==(100,100) and np.all(HS.sum(axis=1)==22)
assert np.array_equal(HS@HS,16*np.eye(100,dtype=np.int64)-6*HS+6*np.ones((100,100),dtype=np.int64))
results['HS']={'strongly_regular_identity_exact':True,'minimal_adjacency_eigenvalue':-8,'q_min':14}
results['elapsed_seconds']=time.time()-start;results['status']='all independent checks passed'
(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8');log('SUCCESS',results['elapsed_seconds'],'seconds')
