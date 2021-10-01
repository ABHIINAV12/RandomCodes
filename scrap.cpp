// -----------------------------------------------------------------------------------------------------------------------------------

// sparse table implementation

#include <bits/stdc++.h>
using namespace std;
#define ll long long int
#define mod 1000000007

int sparse[100005][20];
int Log[100005];
int A[100005];

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
	int t,k,q;
	t=1;
	while(t--){
	    int n,m;
	    cin>>n;
	    Log[1] = 0;
        for (int i = 2; i <= 100005; i++){
            Log[i] = Log[i/2] + 1;
        }
	    for(int i=0;i<n;i++){
	        cin>>A[i];
	    }
	    k = Log[n];
	    for(int i=0;i<n;i++){
            sparse[i][0] = A[i];
        }
        for(int j=1;j<=k;j++){
            for(int i=0;i+(1<<j)<=n;i++){
                sparse[i][j] = max(sparse[i][j-1],sparse[i+(1<<(j-1))][j-1]);
            }
        }
        
	    int x,y;
	    cin>>m>>x>>y;
	    int l = min(x,y);
	    int r = max(x,y);
	    int j = Log[r-l+1];
	    ll sum =max(sparse[l][j],sparse[r-(1<<j)+1][j]);
	    for(int i=1;i<m;i++){
	        x=(x+7);
		    while(x>=n-1)x-=n-1;
            y=(y+11);
            while(y>=n)y-=n;
	        l = min(x,y);
	        r = max(x,y);
	        j = Log[r-l+1];
	        sum+= max(sparse[l][j],sparse[r-(1<<j)+1][j]);
	    }
	    cout<<sum<<"\n";
	}
	return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// max on sparese tree with binary search

#include<bits/stdc++.h>
using namespace std;
using ll = long long;

#define F first
#define S second

int main(){
    ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
    ll n; cin >> n;
    vector<ll> A(n);
    for(int i = 0; i < n; i++) cin >> A[i];
    ll q; cin >> q;
    vector<ll> pref(n-1);
    for(int i = 0; i < n-1; i++){
        pref[i] = A[i+1] - A[i]; 
    }
    ll k = log2(n-1);
    ll st[n-1][k+1];
    for(int i = 0; i < n-1; i++) st[i][0] = pref[i];
    ll log[n];
    log[1] = 0;
    for(int i = 2; i < n; i++) log[i] = log[i/2] + 1;
    for(int j = 1; j <= k; j++){
        for(int i = 0; i + (1ll << j) <= n-1; i++){
            st[i][j] = max(st[i][j-1], st[i+ (1ll << (j-1))][j-1]);
        }
    }
    while(q--){
        ll t, d; cin >> t >> d;
        ll k = n, left = 0, right = n-1;
        while(left <= right){
            ll mid = (left + right) >> 1;
            if(A[mid] <= t) left = mid + 1;
            else right = mid - 1, k = mid;
        }
        k--; ll ans;
        if(k == 0) {
            ans = 0;
        } else {
            ans = k;
            ll l = 0, r = k-1, last = k - 1;
            while(l <= r){
                ll mid = l + ((r - l) >> 1);
                ll j = log[last - mid + 1];
                ll mx = max(st[mid][j], st[last-(1ll << j) + 1][j]);
                if(mx <= d){
                    r = mid-1;
                    ans = mid;
                } else {
                    l = mid+1;
                }
            }
        }
        cout << ans + 1 << '\n';
    }
    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// inoi  3*n tilling
// editorial : https://discuss.codechef.com/t/3xn-tiling-editorial/82065

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef vector<int> vi;
typedef vector<vector<int>> vivi;
typedef pair<int,int> pi;
typedef vector<int> vi;
typedef pair<int,int> pi;
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define inset(s,i) s.find(i)!=s.end()
const int MAX=1000001;
ll dp2[MAX],dp31[MAX],dp32[MAX],dp[MAX];
int MOD=1e9+7;
int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    int t;
    cin>>t;
    while(t--){
        int k,n;
        cin>>k>>n;
        dp2[1]=0;
        dp2[2]=dp2[3]=1;
        dp31[1]=dp31[2]=dp32[1]=dp32[2]=dp32[3]=0;
        dp31[3]=dp[1]=dp[2]=1;
        dp[3]=2;
        for(int i=4;i<=1e6;i++){
           dp2[i]=(dp2[i-2]+dp2[i-3])%MOD;
           dp31[i]=(dp[i-3]+dp31[i-3])%MOD;
           dp32[i]=(dp31[i-1]+dp32[i-3])%MOD;
           dp[i]=(dp[i-1]+2*dp32[i-2]+dp[i-3])%MOD;

        }
        if(k==1){
            if(n%3==0){cout<<1<<endl;}
            else{cout<<0<<endl;}
        }
        if(k==2){
            cout<<dp2[n]<<endl;
        }
        if(k==3){
            cout<<dp[n]<<endl;
        }

    }
    
}
    
// -----------------------------------------------------------------------------------------------------------------------------------


// problem : free ticket

#include <bits/stdc++.h>
using namespace std;
#define fast ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
#define ll long long int
#define tc ll test;cin >> test;while(test--)
#define vi vector<ll>
#define pll pair<ll,ll>
#define pb push_back
#define mp make_pair
#define INF 1e18
#define MOD 1e9 + 7
#define ff first
#define ss second
#define in >>
#define out <<
#define space << " " <<
#define spacef << " "
#define fo(i,a,b) for(ll i = a; i <= b; i++)
#define nextline out "\n"
#define print(x) for(auto i : x ) cout out i spacef
#define mmax(x,i) x = max(x,i)
#define mmin(x,i) x = min(x,i)
#define N 235

vector<pll> adj[N];
ll n,m;

ll dijkstra(ll s){

    vi dist(N,INF);
    priority_queue<pll> q;
    vector<bool> visited(N,false);

    dist[s] = 0;
    q.push({0,s});

    while(!q.empty()){
        ll a = q.top().second;
        q.pop();

        if(visited[a] == true) continue;
        visited[a] = true;

        for(auto u : adj[a]){
            ll b = u.first;
            ll w = u.second;

            if(dist[a] + w < dist[b]){
                dist[b] = dist[a] + w;
                q.push({-dist[b],b});
            }
        }
    }

    ll maxi = -INF;

    fo(i,1,n) if(dist[i] != INF) mmax(maxi,dist[i]);

    return maxi;

}

int main() {

    fast;

    cin in n in m;

    fo(i,1,m){
        ll a,b,c;
        cin in a in b in c;

        adj[a].pb(mp(b,c));
        adj[b].pb(mp(a,c));
    }

    ll maxweight = -INF;

    fo(i,1,n) mmax(maxweight , dijkstra(i));

    cout out maxweight;

    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// problem : inoi triathon

#include <bits/stdc++.h>
using namespace std;
#define fast ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
#define ll long long
#define tc ll test;cin >> test;while(test--)
#define vi vector<ll>
#define pb push_back
#define mp make_pair
#define INF 0x3f3f3f3f3f
#define MOD 1e9 + 7
#define ff first
#define ss second
#define in >>
#define out <<
#define space << " " <<
#define spacef << " "
#define fo(i,a,b) for(ll i = a; i <= b; i++)
#define nextline out "\n"
#define print(x) for(auto i : x ) cout out i spacef
#define mmax(x,i) x = max(x,i)
#define mmin(x,i) x = min(x,i)

int main() {

    ll n;
    cin in n;

    vi cobol(n+5);
    vector<vi> other(n+5,vi(2));

    fo(i,0,n-1){
        ll x,y;
        cin in cobol[i] in x in y;
        other[i][0] = x+y;
        other[i][1] = i;
    }

    sort(other.rbegin(),other.rend());

    ll waiting = 0;

    ll ans = -INF;

    fo(i,0,n-1){
        ll time = waiting + other[i][0] + cobol[other[i][1]];
        waiting += cobol[other[i][1]];
        mmax(ans,time);
    }

    cout out ans;

    return 0;

}
// -----------------------------------------------------------------------------------------------------------------------------------

// problem : calvins game

#include <bits/stdc++.h>
using namespace std;
#define fast ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
#define ll long long
#define tc ll test;cin >> test;while(test--)
#define vi vector<ll>
#define pb push_back
#define mp make_pair
#define INF 0x3f3f3f3f3f
#define MOD 1e9 + 7
#define ff first
#define ss second
#define in >>
#define out <<
#define space << " " <<
#define spacef << " "
#define fo(i,a,b) for(ll i = a; i <= b; i++)
#define nextline out "\n"
#define print(x) for(auto i : x ) cout out i spacef
#define mmax(x,i) x = max(x,i)
#define mmin(x,i) x = min(x,i)

int main() {

    ll n,start;
    cin in n in start;

    vi a(n+5);
    fo(i,1,n) cin in a[i];

    vi forwarddp(n+5,-INF);
    vi backwarddp(n+5);

    forwarddp[start] = 0;
    if(start != n) forwarddp[start+1] = a[start+1];

    fo(i,start+2,n) forwarddp[i] = a[i] + max(forwarddp[i-1] , forwarddp[i-2]);

    backwarddp[n] = forwarddp[n];
    backwarddp[n-1] = max(forwarddp[n]+a[n-1],forwarddp[n-1]);

    for(ll i = n-2; i >= 1; i--) backwarddp[i] = max(forwarddp[i],a[i]+max(backwarddp[i+1],backwarddp[i+2]));

    cout out backwarddp[1];

    return 0;

}


// -----------------------------------------------------------------------------------------------------------------------------------


// problem : sequene land

#include <bits/stdc++.h>
using namespace std;
#define fast ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
#define ll long long int
#define tc ll test;cin >> test;while(test--)
#define vi vector<ll>
#define pll pair<ll,ll>
#define pb push_back
#define mp make_pair
#define INF 1e18
#define MOD 1e9 + 7
#define ff first
#define ss second
#define in >>
#define out <<
#define space << " " <<
#define spacef << " "
#define fo(i,a,b) for(ll i = a; i <= b; i++)
#define nextline out "\n"
#define print(x) for(auto i : x ) cout out i spacef
#define mmax(x,i) x = max(x,i)
#define mmin(x,i) x = min(x,i)
#define N 305

vector<bool> visited(N,false);
ll cnt = 0;
vi adj[N];

void dfs(ll s){

    if(visited[s] == true) return;
    visited[s] = true;

    cnt++;

    for(auto u : adj[s]) dfs(u);
}

int main() {

    fast;

    ll n,k;
    cin in n in k;

    vi id[n+5];

    fo(i,1,n){
        ll l;
        cin in l;

        while(l--){
            ll x;
            cin in x;
            id[i].push_back(x);
        }
    }

    fo(i,1,n) sort(id[i].begin(),id[i].end());

    fo(i,1,n){
        fo(j,1,n){
            if(i == j) continue;

            ll common = 0;

            for(auto u : id[i]) if(binary_search(id[j].begin(),id[j].end(),u) == true) common++;

            if(common >= k){
                adj[i].pb(j);
                adj[j].pb(i);
            }
        }
    }

    dfs(1);

    cout out cnt;

    return 0;
}


// -----------------------------------------------------------------------------------------------------------------------------------

// problem : wealth disparity

#include <bits/stdc++.h>
using namespace std;
#define fast ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
#define ll long long int
#define tc ll test;cin >> test;while(test--)
#define vi vector<ll>
#define pll pair<ll,ll>
#define pb push_back
#define mp make_pair
#define INF 1e18
#define MOD 1e9 + 7
#define ff first
#define ss second
#define in >>
#define out <<
#define space << " " <<
#define spacef << " "
#define fo(i,a,b) for(ll i = a; i <= b; i++)
#define nextline out "\n"
#define print(x) for(auto i : x ) cout out i spacef
#define mmax(x,i) x = max(x,i)
#define mmin(x,i) x = min(x,i)
#define N 100005

ll ans = -INF;
vi adj[N];
vi wealth(N);

ll dfs(ll s){

    ll minwealth = wealth[s];

    for(auto u : adj[s]) mmin(minwealth,dfs(u));
    mmax(ans,wealth[s]-minwealth);

    return minwealth;
}

int main() {

    fast;

    ll n;
    cin in n;

    ll root;

    fo(i,1,n) cin in wealth[i];
    fo(i,1,n) {
        ll x;
        cin in x;

        if(x != -1) adj[x].pb(i);
        if(x == -1) root = i;
    }

    dfs(root);
    cout out ans;

    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// problem : Table sum, lazy add with max

#include<bits/stdc++.h>
using namespace std;

#define rep(i, a, b) for(int i=a; i<=b; i++)

int n;
int a[200001];
int mx[800001];
int op[800001];

void push(int x, int lx, int rx){
  if(!op[x] || rx==lx+1) return;
  op[2*x+1]+=op[x];
  op[2*x+2]+=op[x];
  mx[2*x+1]+=op[x];
  mx[2*x+2]+=op[x];
  op[x]=0;
}

void add(int l, int r, int v, int x, int lx, int rx){
  if(lx>=r || rx<=l) return;
  push(x, lx, rx);
  if(lx>=l && rx<=r){
    op[x]=v; //== +=v
    mx[x]+=v;
    push(x, lx, rx);
    return;
  }
  int m=(lx+rx)/2;
  add(l, r, v, 2*x+1, lx, m);
  add(l, r, v, 2*x+2, m, rx);
  mx[x]=max(mx[2*x+1], mx[2*x+2]);
}

int get_max(int l, int r, int x, int lx, int rx){
  push(x, lx, rx);
  if(lx>=l && rx<=r) return mx[x];
  if(lx>=r || rx<=l) return -2e9;
  int m=(lx+rx)/2;
  return max(get_max(l, r, 2*x+1, lx, m), get_max(l, r, 2*x+2, m, rx));
}

int main(){
  cin.tie(0)->sync_with_stdio(0);
  // freopen("input.txt", "r", stdin);
  // freopen("output.txt", "w", stdout);

  cin >> n;
  rep(i, 1, n){
    cin >> a[i];
    add(i, i+1, a[i]+i, 0, 0, n+1);
  }

  cout << get_max(1, n+1, 0, 0, n+1) << " ";

  rep(i, 0, n-2){
    add(1, n+1, 1, 0, 0, n+1);
    add(n-i, n-i+1, -n, 0, 0, n+1);
    cout << get_max(1, n+1, 0, 0, n+1) << " ";
  }

}

// can you hear the silence? can you see the dark? can you feel my heart?
// range add, max

// -----------------------------------------------------------------------------------------------------------------------------------

// problem : Highway bypass

#include <bits/stdc++.h>
using namespace std;
#define fast ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
#define ll long long
#define tc ll test;cin >> test;while(test--)
#define vi vector<ll>
#define pb push_back
#define mp make_pair
#define INF 0x3f3f3f3f3f
#define MOD 20011
#define ff first
#define ss second
#define in >>
#define out <<
#define space << " " <<
#define spacef << " "
#define fo(i,a,b) for(ll i = a; i <= b; i++)
#define nextline out "\n"
#define print(x) for(auto i : x ) cout out i spacef
#define mmax(x,i) x = max(x,i)
#define mmin(x,i) x = min(x,i)

int main() {

    fast;

    ll n,m,d;
    cin in n in m in d;

    vector<vi> grid(n+5,vi(m+5));

    fo(i,1,n){
        fo(j,1,m){
            cin in grid[i][j];
        }
    }

    vector<vector<vi>> dp(n+5,vector<vi>(m+5,vi(2,0)));

    // SUB-PROBLEM :

    // dp[i][j][1] = Number of ways to come to (i,j) horizontally
    // dp[i][j][2] = Number of ways to come to (i,j) vertically

    // BASE CASE :

    dp[1][1][1] = 1;
    dp[1][1][2] = 1;

    fo(j,2,d+1){
        if(grid[1][j] == 0) break;
        dp[1][j][1] = 1;
    }

    fo(i,2,d+1){
        if(grid[i][1] == 0) break;
        dp[i][1][2] = 1;
    }

    // RECURRENCE RELATION :

    fo(i,2,n){
        fo(j,2,m){
            if(grid[i][j] == 0) continue;
            for(ll k = i-1; k >= max(1ll,i-d); k--){
                if(grid[k][j] == 0) break;
                dp[i][j][2] += dp[k][j][1];
                dp[i][j][2] %= MOD;
            }
            for(ll k =j-1; k >= max(1ll,j-d); k--){
                if(grid[i][k] == 0) break;
                dp[i][j][1] += dp[i][k][2];
                dp[i][j][1] %= MOD;
            }
        }
    }

    // OPTIMAL ANSWER :

    cout out (dp[n][m][1]%MOD + dp[n][m][2]%MOD)%MOD;

    return 0;

}

// -----------------------------------------------------------------------------------------------------------------------------------

// problem : special sums

#include <bits/stdc++.h>
using namespace std;
#define fast ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
typedef long long int ll;
#define tc ll test;cin >> test;while(test--)
#define vi vector<ll>
#define pb push_back
#define mp make_pair
#define pll pair<ll,ll>
#define INF 1e18
#define MOD 1000000007
#define ff first
#define ss second
#define in >>
#define out <<
#define space << " " <<
#define spacef << " "
#define fo(i,a,b) for(ll i = a; i <= b; i++)
#define nextline out "\n"
#define print(x) for(auto i : x ) cout out i spacef
#define mmax(x,i) x = max((x),(i))
#define mmin(x,i) x = min((x),(i))
#define N 105

int main() {

    fast;

    ll n;
    cin in n;

    ll ans = -INF;

    vi a(n+5);
    vi b(n+5);
    vi presum(n+5);
    vi diff1(n+5,-INF);
    vi diff2(n+5,-INF);

    fo(i,1,n) {
        cin in a[i];
        mmax(ans,a[i]);
    }
    fo(i,1,n) {
        cin in b[i];
        presum[i] = presum[i-1] + b[i];
        diff1[i] = max(diff1[i-1] , a[i] - presum[i]);
        diff2[i] = max(diff2[i-1] , a[i] + presum[i-1]);
    }

    // i...j

    fo(i,1,n) mmax(ans,a[i] + presum[i-1] + diff1[i-1]);

    // j...i

    fo(i,1,n) mmax(ans , presum[n] - presum[i] + a[i] + diff2[i-1]);

    cout out ans;

    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// problem : periodic strings

#include <bits/stdc++.h>
using namespace std;
#define fast ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);
#define ll long long int
#define tc ll test;cin >> test;while(test--)
#define vi vector<ll>
#define pb push_back
#define mp make_pair
#define INF 0x3f3f3f3f3f
#define MOD 13371337
#define ff first
#define ss second
#define in >>
#define out <<
#define space << " " <<
#define spacef << " "
#define fo(i,a,b) for(ll i = a; i <= b; i++)
#define nextline out "\n"
#define print(x) for(auto i : x ) cout out i spacef
#define mmax(x,i) x = max(x,i)
#define mmin(x,i) x = min(x,i)
#define N 200005

ll power(ll x, ll y , ll m){
    if(y == 1) return x;

    ll res = (power(x, y / 2 , m)) % m;

    if(y % 2) return ((res * res * x) % m);
    return ((res * res) % m);
}

int main() {

    fast;

    ll n,m;
    cin in n in m;

    vi dp(n+5);

    // dp[i] = number of ways we can create non-perioidic strings with length n

    dp[1] = 2;

    fo(i,2,n){

        dp[i] = power(2,i,m) - dp[1];
        dp[i] %= m;

        for(ll j = 2; j*j <= i; j++){
            if(i%j == 0){
                dp[i] -= dp[j];

                dp[i] += m;
                dp[i] %= m;

                if(j*j != i ){
                    dp[i] -= dp[i/j];
                    dp[i] += m;
                    dp[i] %= m;
                }
            }
        }
    }

    cout out dp[n];

    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// problem : Brackets inoi

#include <bits/stdc++.h>
using namespace std;
#define fast ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL)
#define ll long long int

int main() {

    fast;

    ll n,k;
    cin >> n >> k;

    vector<ll> value(n+5);
    vector<ll> bracket(n+5);
    
    for(ll i = 1; i <= n; i++) cin >> value[i];
    for(ll i = 1; i <= n; i++) cin >> bracket[i];

    vector<vector<ll>> dp(n+5,vector<ll>(n+5));

    // dp[i][j] = maximum sum possible of well bracketed sequence if we consider only from i to j ( both inclusive )

    // base case :

    for(ll i = 1; i <= n; i++){
        for(ll j = 1; j <= i-1; j++) {
            dp[i][j] = -1;
        }
    }
    for(ll i = 1; i <= n; i++) dp[i][i]= 0;
    for(ll i = 1; i <= n-1 ; i++){
        if(bracket[i+1]-bracket[i] == k){
            dp[i][i+1] = value[i] + value[i+1];
        }
    }

    // recurrence :

    for(ll diff = 2; diff <= n-1; diff++){
        
        ll i = 1;
        ll j = 1+diff;

        while(j <= n){

            if(bracket[j]-bracket[i] == k){
                dp[i][j] = value[i]+value[j]+dp[i+1][j-1];
            }
            else{
                dp[i][j] = max(dp[i][j-1],dp[i+1][j]);
            }

            for(ll m = i+1; m <= j; m++){
                dp[i][j] = max(dp[i][j],dp[i][m-1]+dp[m][j]);
            }

            i++;
            j++;
        }
    }

    // answer :

    cout << dp[1][n];

    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// problem : https://www.codechef.com/INOIPRAC/problems/ROADTRIP

#pragma GCC optimize("O3")
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;
#define ordered_set     tree<int, null_type,less<int>, rb_tree_tag,tree_order_statistics_node_update> 
// find_by_order(k):    It returns to an iterator to the kth element (counting from zero) in the set
// order_of_key(k) :    It returns to the number of items that are strictly smaller than our item k
#define fast            ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0)
#define pb              push_back
#define f               first
#define s               second
#define F(i,a,b)        for(i=a;i<b;i++)
#define nl              "\n"
#define sp              " " 
#define all(c)          (c).begin(),(c).end()
#define rev(i,b,a)      for(int i=b;i>=a;i--)
#define iota            cout<<-1<<nl
#define cty             cout<<"YES"<<nl
#define ctn             cout<<"NO"<<nl
#define lmax            LLONG_MAX
#define lmin            LLONG_MIN
#define sz(v)           (v).size()
#define deci(n)         fixed<<setprecision(n)
#define c(x)            cout<<(x)
#define csp(x)          cout<<(x)<<" "
#define c1(x)           cout<<(x)<<nl
#define c2(x,y)         cout<<(x)<<" "<<(y)<<nl
#define c3(x,y,z)       cout<<(x)<<" "<<(y)<<" "<<(z)<<nl
#define c4(a,b,c,d)     cout<<(a)<<" "<<(b)<<" "<<(c)<<" "<<(d)<<nl
#define c5(a,b,c,d,e)   cout<<(a)<<" "<<(b)<<" "<<(c)<<" "<<(d)<<" "<<(e)<<nl
#define c6(a,b,c,d,e,f) cout<<(a)<<" "<<(b)<<" "<<(c)<<" "<<(d)<<" "<<(e)<<" "<<(f)<<nl
typedef long double     ld;
typedef long long       ll;
typedef vector<ll>      vll;
typedef pair<ll,ll>     pll;
typedef vector<pll>     vpll;
const int mod=998244353;
const int mod1=1000000007;
const double pi=3.14159265358979323846264338327950288419716939937510582097494459230;
// priority_queue<t>
ll max(ll a , ll b){ if(a>b)return a; return b;}
ll min(ll a , ll b){ if(a<b)return a; return b;}
ll cdiv(ll a, ll b) { return a/b+((a^b)>0&&a%b); } // divide a by b rounded up
ll fdiv(ll a, ll b) { return a/b-((a^b)<0&&a%b); } // divide a by b rounded down
ll pct(ll x) { return __builtin_popcount(x); } // # of bits set
ll poww(ll a, ll b) { ll res=1; while(b) { if(b&1)  res=(res*a); a=(a*a); b>>=1; } return res; }
ll modI(ll a, ll m=mod) { ll m0=m,y=0,x=1;  if(m==1) return 0;  while(a>1)  { ll q=a/m; ll t=m;  m=a%m;  a=t; t=y; y=x-q*y;  x=t; } if(x<0) x+=m0; return x;}
ll powm(ll a, ll b,ll m=mod) {ll res=1; while(b) {  if(b&1) res=(res*a)%m;  a=(a*a)%m; b>>=1;   }   return res;}
//*******************************************************************************************************************************************//
vll v[1000001];
bool vis[1000001];
ll ans;
ll a[1000001];
void dfs(ll s)
{
    ans+=a[s];
    for(auto u:v[s])
        if(!vis[u])
        vis[u]=1,dfs(u);
}
void iamzeus()
{ ll i,j,x=0,y,n,m,d,cnt=0,z,c=2,k;
 cin>>n>>m>>k;
 for(i=1;i<=n;i++)
     v[i].clear(),vis[i]=0;
 for(i=1;i<=m;i++)
 {
     cin>>x>>y;
     v[x].pb(y),v[y].pb(x);
 }
 for(i=1;i<=n;i++)
     cin>>a[i];
 vll temp;
 for(i=1;i<=n;i++)
     if(!vis[i])
     ans=0,vis[i]=1,dfs(i),temp.pb(ans);
 if(sz(temp)<k)
 {
     iota;
     return;
 }
 sort(all(temp));
 ans=0;
 i=0,j=sz(temp)-1;
 z=0;
 while(z<k)
 {
     if(z%2==0)
         ans+=temp[j],j--;
     else
         ans+=temp[i],i++;
     z++;
 }
 c1(ans);
};
int main()
{fast;
    int t;
//  t=1;
cin >>t;
 while(t--)
     iamzeus();
}

// -----------------------------------------------------------------------------------------------------------------------------------

// problem : https://www.codechef.com/INOIPRAC/problems/TINOI17A


#include<bits/stdc++.h>
using namespace std;

typedef long long int                  ll; 
typedef pair<int,int>                  pii;
#define fi                             first
#define se                             second
#define INF                            0x3f3f3f3f
#define MOD                            1000000007
#define For(i,b)                       for(int i=0;i<b;i++)
#define FoR(i,a,b)                     for(int i=a;i>=b;i--)
#define For1(i,b)                      for(int i=1;i<=b;i++)
#define FOR(i,a,b)                     for(int i=a;i<=b;i++)
#define MS0(X)                         memset((X), 0, sizeof((X)))
#define MS1(X)                         memset((X), -1, sizeof((X)))
#define REMAX(a,b)                     (a)=max((a),(b)) // set a to the maximum of a and b
#define REMIN(a,b)                     (a)=min((a),(b));

#define fast_io                        ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);  srand(time(NULL));
#define sublimeProblem                 freopen("input.txt", "r", stdin); freopen("output.txt", "w", stdout); freopen("debug_log.txt", "w", stderr);

#define N 100005
int n,r,c,a,b;

set<pii> g,visited;

pii dfs(pii u){
    if(visited.count(u) || g.count(u)==0 ||u.fi<1||u.se<1||u.fi>r||u.se>c)
        return {0,0};

    visited.insert(u);

    int cc=1,adj=0;
    adj+=g.count({u.fi,u.se+1});
    adj+=g.count({u.fi,u.se-1});
    adj+=g.count({u.fi+1,u.se});
    adj+=g.count({u.fi-1,u.se});

    pii t;
    
    t=dfs({u.fi,u.se+1});
    cc+=t.fi; adj+=t.se;

    t=dfs({u.fi,u.se-1});
    cc+=t.fi; adj+=t.se;

    t=dfs({u.fi+1,u.se});
    cc+=t.fi; adj+=t.se;

    t=dfs({u.fi-1,u.se});
    cc+=t.fi; adj+=t.se;

    return {cc,adj};
}

void solve(){
    cin>>r>>c>>n;
    For(i,n){
        cin>>a>>b;
        g.insert({a,b});
    }    

    int mx=-1;
    for(auto i:g){
        if(visited.count(i))
            continue;
        pii returnValue=dfs(i);
        REMAX(mx,(4*returnValue.fi-returnValue.se));
    }
    cout<<mx;
}

int main(){

   // sublimeProblem;
    fast_io;

    // int t; cin>>t; while(t--)
    { 
        solve();   
    }

}

// -----------------------------------------------------------------------------------------------------------------------------------

// problem : https://www.codechef.com/INOIPRAC/problems/INOI2001


#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define pb push_back
#define forp(i,a,b) for(ll i=a;i<=b;i++)
#define forn(i,a,b) for(ll i=a;i>=b;i--)
#define read(a,n) forp(i,1,n){cin>>a[i];}
#define vi vector<ll>
#define newl '\n'
#define mod 1000000007
#define ml map<ll,ll>
#define form(m,it) for(auto it=m.begin();it!=m.end(); it++)
#define ff first
#define ss second

void solve(){
    ll n,m; cin>>n>>m;
    vi v[n+1];
    ll x,y;
    forp(i,1,m){
        cin>>x>>y;
        v[x].pb(y);
        v[y].pb(x);
    }
    queue<ll> q;
    ll a; 
    bool vis1[n+1]={0},vis2[n+1]={0};
    ll sz1[n+1]={-1},sz2[n+1]={-1};
    ll level1[n+1]={0},level2[n+1]={0};
    ll sum1[n+1]={0},sum2[n+1]={0};
    forp(i,1,n){
        if(!vis1[i]){
            q.push(i);
            vis1[i]=1;
            sz1[i]=0;
            level1[i]=1;
            while(!q.empty()){
                a=q.front();
                sum1[i]+=level1[a];
                q.pop();
                sz1[i]++;
                for(auto u:v[a]){
                    if(!vis1[u]){
                        vis1[u]=1;
                        level1[u]=level1[a]+1;
                        q.push(u);
                    }
                }
            }
        }
    }
    forn(i,n,1){
        if(!vis2[i]){
            q.push(i);
            vis2[i]=1;
            sz2[i]=0;
            level2[i]=1;
            while(!q.empty()){
                a=q.front();
                sum2[i]+=level2[a];
                q.pop();
                sz2[i]++;
                for(auto u:v[a]){
                    if(!vis2[u]){
                        vis2[u]=1;
                        level2[u]=level2[a]+1;
                        q.push(u);
                    }
                }
            }
        }
    }
    ll te=0,to=0;
    forp(i,1,n){
        if(sz1[i]!=-1&&sz1[i]%2==0){
            te+=(sum1[i]);
        }
        if(sz2[i]!=-1&&sz2[i]%2==1){
            to+=(sum2[i]);
        }
    }
    cout<<te<<" "<<to<<newl;
}

int main() {
    clock_t start,end;
    start=clock();
    ios_base::sync_with_stdio(false); 
    cin.tie(NULL); 
    cout.tie(NULL);
    #ifndef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
    #endif
    ll t=1;
    cin>>t;
    while(t--){
        solve();
    }
    end=clock();
    //cout<<newl<<"Time : "<<double(end - start)/double(CLOCKS_PER_SEC)<<newl;
}


// -----------------------------------------------------------------------------------------------------------------------------------

// problem : https://www.codechef.com/INOIPRAC/problems/TWOPATHS

#include <bits/stdc++.h>
using namespace std;
using ll = long long;
using vi = vector<int>;
using vvi = vector<vi>;
using vl = vector<ll>;
using vvl = vector<vl>;
using vvvi = vector<vvi>;
int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);

    int t,n = 1001,m = 1001,k = 21;
    vvvi dp1(n,vvi(m,vi(k))),dp2(n,vvi(m,vi(k)));
    vvi g(n,vi(m));
    vvi p(n,vi(m));
    cin>>t;

    while (t--){
        cin>>n>>m>>k;
        for (int i = 1;i<=n;i++){
            for (int j = 1;j<=m;j++){
                cin>>g[i][j];
                p[i][j] = g[i][j];
                if (j >= 1)p[i][j] += p[i][j - 1];
            }
        }
        for (int i = 1;i<=n;i++){
            for (int j = 1;j<=m;j++){
                for (int l = 0;l<=k;l++){
                    dp1[i][j][l] = dp1[i - 1][j][l];
                    dp2[i][j][l] = dp2[i - 1][j][l];
                    if (l >= 1)dp1[i][j][l] = max(dp1[i][j][l],dp1[i - 1][j - 1][l - 1]);
                    if (l >= 1)dp2[i][j][l] = min(dp2[i][j][l],dp2[i - 1][j - 1][l - 1]);
                    dp1[i][j][l] += p[i][j];
                    dp2[i][j][l] += p[i][j];
                }
            }
        }
        int ans = -1e9;
        for (int a = 0;a<=m;a++){
            for (int b = a + k + 2;b<=m;b++)ans = max(ans,dp1[n][b][k] - dp2[n][a][k]);
        }
        cout<<ans<<'\n';
    }
    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Problem : https://www.codechef.com/INOIPRAC/problems/AMONGUS2

#include "bits/stdc++.h"
using namespace std;

#define rep(i,a,b) for(int i=a; i<=b; i++)
#define nl '\n'
#define f first
#define s second
typedef pair<int, int> pii;

int cnt[2];
int t, n, q;
vector<pii> adj[500001];
int col[500001];
bool imp;

void dfs(int x){
  cnt[col[x]]++;
  for(pii p:adj[x]){
    int c=(col[x]+p.s)%2;
    if(col[p.f]!=-1){
      if(col[p.f]!=c){
        imp=1;
        return;
      }
      continue;
    }
    col[p.f]=c;
    dfs(p.f);
    if(imp) return;
  }  
}

int main(){
  cin.tie(0)->sync_with_stdio(0);
  // freopen("input.txt", "r", stdin);
  // freopen("output.txt", "w", stdout);

  cin >> t;
  while(t--){
    cin >> n >> q;
    rep(i, 1, n){
      adj[i].clear();
      col[i]=-1;
    }
    while(q--){
      int op,a, b; cin >> op >> a >> b;
      adj[a].push_back({b, op%2});
      adj[b].push_back({a, op%2});
    }

    imp=0;
    int ans=0;
    rep(i, 1, n){
      if(col[i]!=-1) continue;
      cnt[0]=cnt[1]=0;
      col[i]=0;
      dfs(i);
      if(imp) break;
      ans+=max(cnt[0], cnt[1]);
    }
    if(imp) cout << -1 << nl;
    else cout << ans << nl;
  }
}

// -----------------------------------------------------------------------------------------------------------------------------------


// Problem : https://www.codechef.com/INOIPRAC/problems/TINOI17B
// editorial : https://cs.stackexchange.com/questions/86107/inoi-2017-problem-2-training 

#include "bits/stdc++.h"
using namespace std;

const int N = 5000 + 50;
const long long INF = (1LL << 60LL);

int n;
long long s[N], e[N];
long long dp[N][N];

inline long long cube(long long x) {
    return x * x * x;
}

inline long long sum(long long x) {
    if (x == 0) {
        return 0;
    }
    return (x % 10LL) + sum(x / 10LL);
}

int main() {
    cin >> n >> s[0];
    for (int i = 1; i <= n; i++) {
        s[i] = s[i - 1] + cube(sum(s[i - 1]));
    }
    for (int i = 1; i <= n; i++) {
        cin >> e[i];
    }
    // dp[i][j] = Max XP you can attain by training exactly j times among gyms labelled 1..i
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            dp[i][j] = -INF;
        }
    }
    dp[0][0] = 0;
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= i; j++) {
            dp[i][j] = dp[i - 1][j] + (s[j] * e[i]); // Battle at the (i)th gym
            if (j > 0) {
                dp[i][j] = max(dp[i][j], dp[i - 1][j - 1]); // Train at the (i)th gym
            }
        }
    }
    long long ans = 0;
    for (int i = 0; i <= n; i++) {
        ans = max(ans, dp[n][i]);
    }
    cout << ans << "\n";
}

// -----------------------------------------------------------------------------------------------------------------------------------

//Problem : Geothermals chef 2 chef solution
// Editorial : https://www.youtube.com/watch?v=xUhfWFRe024


#include "bits/stdc++.h"
#pragma GCC optimize ("O3")
#pragma GCC target ("sse4")
 
using namespace std;
 
typedef long long ll;
typedef long double ld;
typedef complex<ld> cd;
 
typedef pair<int, int> pi;
typedef pair<ll,ll> pl;
typedef pair<ld,ld> pd;
 
typedef vector<int> vi;
typedef vector<ld> vd;
typedef vector<ll> vl;
typedef vector<pi> vpi;
typedef vector<pl> vpl;
typedef vector<cd> vcd;

template<class T> using pq = priority_queue<T>;
template<class T> using pqg = priority_queue<T, vector<T>, greater<T>>;
 
#define FOR(i, a, b) for (int i=a; i<(b); i++)
#define F0R(i, a) for (int i=0; i<(a); i++)
#define FORd(i,a,b) for (int i = (b)-1; i >= a; i--)
#define F0Rd(i,a) for (int i = (a)-1; i >= 0; i--)
#define trav(a,x) for (auto& a : x)
#define uid(a, b) uniform_int_distribution<int>(a, b)(rng)
 
#define sz(x) (int)(x).size()
#define mp make_pair
#define pb push_back
#define f first
#define s second
#define lb lower_bound
#define ub upper_bound
#define all(x) x.begin(), x.end()
#define ins insert

template<class T> bool ckmin(T& a, const T& b) { return b < a ? a = b, 1 : 0; }
template<class T> bool ckmax(T& a, const T& b) { return a < b ? a = b, 1 : 0; }
 
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

void __print(int x) {cerr << x;}
void __print(long x) {cerr << x;}
void __print(long long x) {cerr << x;}
void __print(unsigned x) {cerr << x;}
void __print(unsigned long x) {cerr << x;}
void __print(unsigned long long x) {cerr << x;}
void __print(float x) {cerr << x;}
void __print(double x) {cerr << x;}
void __print(long double x) {cerr << x;}
void __print(char x) {cerr << '\'' << x << '\'';}
void __print(const char *x) {cerr << '\"' << x << '\"';}
void __print(const string &x) {cerr << '\"' << x << '\"';}
void __print(bool x) {cerr << (x ? "true" : "false");}

template<typename T, typename V>
void __print(const pair<T, V> &x) {cerr << '{'; __print(x.first); cerr << ", "; __print(x.second); cerr << '}';}
template<typename T>
void __print(const T &x) {int f = 0; cerr << '{'; for (auto &i: x) cerr << (f++ ? ", " : ""), __print(i); cerr << "}";}
void _print() {cerr << "]\n";}
template <typename T, typename... V>
void _print(T t, V... v) {__print(t); if (sizeof...(v)) cerr << ", "; _print(v...);}
#ifdef DEBUG
#define dbg(x...) cerr << "\e[91m"<<__func__<<":"<<__LINE__<<" [" << #x << "] = ["; _print(x); cerr << "\e[39m" << endl;
#else
#define dbg(x...)
#endif


const int MOD = 1000000007;
const char nl = '\n';
const int MX = 100001; 



struct Seg1 {
ll SZ = 262144; //set this to power of two
ll* seg = new ll[2*SZ]; //segtree implementation by bqi343's Github

ll combine(ll a, ll b) { return max(a, b); }

void build() { F0Rd(i,SZ) seg[i] = combine(seg[2*i],seg[2*i+1]); }

void update(int p, ll value) {  
    for (seg[p += SZ] = value; p > 1; p >>= 1)
        seg[p>>1] = combine(seg[(p|1)^1], seg[p|1]);
}

ll query(int l, int r) {  // sum on interval [l, r]
    ll resL = -1e18, resR = -1e18; r++;
    for (l += SZ, r += SZ; l < r; l >>= 1, r >>= 1) {
        if (l&1) resL = combine(resL,seg[l++]);
        if (r&1) resR = combine(seg[--r],resR);
    }
    return combine(resL,resR);
}
};

struct Seg2 {

ll SZ = 262144; //set this to power of two
ll* seg = new ll[2*SZ]; //segtree implementation by bqi343's Github

ll combine(ll a, ll b) { return min(a, b); }

void build() { F0Rd(i,SZ) seg[i] = combine(seg[2*i],seg[2*i+1]); }

void update(int p, ll value) {  
    for (seg[p += SZ] = value; p > 1; p >>= 1)
        seg[p>>1] = combine(seg[(p|1)^1], seg[p|1]);
}

ll query(int l, int r) {  // sum on interval [l, r]
    ll resL = 1e18, resR = 1e18; r++;
    for (l += SZ, r += SZ; l < r; l >>= 1, r >>= 1) {
        if (l&1) resL = combine(resL,seg[l++]);
        if (r&1) resR = combine(seg[--r],resR);
    }
    return combine(resL,resR);
}
};

Seg1 seg1;
Seg2 seg2;

void solve() {
    int N; cin >> N;

    ll A[N]; F0R(i, N) cin >> A[i];
    ll cur = 0;
    seg1.update(0, 0);
    seg2.update(0, 0);
    FOR(i, 1, N+1) {
        cur += A[i-1];
        seg1.update(i, cur);
        seg2.update(i, cur);
    }

    vpl vals; F0R(i, N) vals.pb({A[i], i});
    sort(all(vals)); reverse(all(vals));
    set<int> used;
    used.ins(-1); used.ins(N);
    ll ans = 0;
    trav(a, vals) {
        int p = a.s;
        auto it = used.lb(p);
        int x1 = *it;
        it--;
        int y1 = *it;
        it++;
        if (x1 != N) {
            it++;
            int x2 = *it;
            it--;
            ckmax(ans, seg1.query(x1+1, x2) - seg2.query(y1+1, p) - A[p] - A[x1]);
            /*if (p == 3) {
                dbg(x1, x2, y1, seg1.query(x1+1, x2), seg2.query(y1+1, p));
            }*/
        }
        if (y1 != -1) {
            it--;
            it--;
            int y2 = *it;
            ckmax(ans, seg1.query(p+1, x1) - seg2.query(y2+1, y1) - A[p] - A[y1]);
        }
        used.ins(p);
    }
    cout << ans << nl;

    F0R(i, N+1) {
        seg1.update(i, -1e18);
        seg2.update(i, 1e18);
    }

}
 
int main() {
    cin.tie(0)->sync_with_stdio(0); 
    cin.exceptions(cin.failbit);
    F0R(i, 262144) {
        seg1.update(i, -1e18);
        seg2.update(i, 1e18);
    }
    int T = 1;
    cin >> T;
    while(T--) {
        solve();
    }

    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------


// cses nested ranges problem

// fuck this man !!
 
#include<bits/stdc++.h> 
#include <iostream> 
#include <vector> 
#include <iterator>
using namespace std;
//#define int long long
#define ld long double
#define pb push_back
#define ff first
#define ss second
#define f(i,x,n) for(int i=x;i<(int)n;++i)
#define vi vector<int>
#define vvi vector<vector<int>>
#define vvvi vector<vector<vector<int>>>
#define pq priority_queue<int>
#define pqs priority_queue<int,vi,greater<int>> 
#define vpii vector<pair<int,int>>
#define pii pair<int,int>
#define all(x) x.begin(),x.end()
#define sz(x) (int)x.size()
#define mpi map<int,int>
#define lb lower_bound //lower_bound returns an iterator pointing to the first element in the range [first,last) which has a value not less than ‘val’. 
#define ub upper_bound // upper_bound returns an iterator pointing to the first element ian the range [first,last) which has a value greater than ‘val’. 
 
 
int mod=1e9+7;
const int mxn=210000;
 
vi vt;
 
int a[mxn],b[mxn];
 
bool fn (array<int,2> &A, array<int,2> &B){
    if(A[0]!=B[0]) return A[0]<B[0];
    return a[A[1]]>a[B[1]];
}
 
struct custom_hash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
 
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitmix64(x + FIXED_RANDOM);
    }
};
 
void st(int curr,int l,int r,int id,int val){
    if(l==r){
        if(l==id) vt[curr]+=val;
        return ;
    }
    if(r<id || id<l) return ;
    int mid=l+r; mid/=2;
    st(2*curr+1,l,mid,id,val);
    st(2*curr+2,mid+1,r,id,val);
    vt[curr]=vt[2*curr+1]+vt[2*curr+2];
}
 
int gt(int curr,int l,int r,int L,int R){
    if(l==r){
        if(l>=L && l<=R) return vt[curr];
        return 0;
    }
    if(R<l || r<L) return 0;
    if(l>=L && r<=R) return vt[curr];
    int mid=l+r; mid/=2;
    return gt(2*curr+1,l,mid,L,R) + gt(2*curr+2,mid+1,r,L,R);
}
 
void solve(){
    int n; cin>>n;
    f(i,0,n) cin>>a[i]>>b[i];
    vi c; f(i,0,n) {
        c.pb(a[i]); c.pb(b[i]);
    }
    sort(all(c));
    int curr=1; unordered_map<int,int,custom_hash> mp; 
    for(auto ii : c){
        if(mp[ii]==0){
            mp[ii]=curr;
            ++curr;
        }
    } ++curr; 
    f(i,0,n){
        a[i]=mp[a[i]];
        b[i]=mp[b[i]];
    }
    vector<array<int,2>> xp;
    f(i,0,n) xp.pb({b[i],i});
    sort(all(xp),fn);
    int x=1; while(x<curr) x*=2;
    f(i,0,2*x-1) vt.pb(0);
    int ans[n]={0};
    for(auto it : xp){
        ans[it[1]]=gt(0,0,x-1,a[it[1]],it[0])>0;
        st(0,0,x-1,a[it[1]],1);
    }
    f(i,0,n) cout<<ans[i]<<" "; cout<<"\n";
    for(auto it : xp) {
        st(0,0,x-1,a[it[1]],-1);
        ans[it[1]]=gt(0,0,x-1,0,a[it[1]])>0;
    }
    f(i,0,n) cout<<ans[i]<<" "; cout<<"\n";
}
 
 
int32_t main(){
    ios_base::sync_with_stdio(false);cin.tie(NULL); 
    srand(time(0));
    #ifndef ONLINE_JUDGE
        freopen("input.txt","r",stdin);
        freopen("output.txt","w",stdout);
    #endif
    int t=1; //cin>>t;
    while(t--) solve();
    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Problem : Sliding Median Cses
// focus this moment !!
 
#include<bits/stdc++.h> 
#include <iostream> 
#include <vector> 
#include <iterator>
using namespace std;
#define int long long
#define ld long double
#define pb push_back
#define ff first
#define ss second
#define f(i,x,n) for(int i=x;i<(int)n;++i)
#define vi vector<int>
#define vvi vector<vector<int>>
#define vvvi vector<vector<vector<int>>>
#define pq priority_queue<int>
#define pqs priority_queue<int,vi,greater<int>> 
#define vpii vector<pair<int,int>>
#define pii pair<int,int>
#define all(x) x.begin(),x.end()
#define sz(x) (int)x.size()
#define mpi map<int,int>
#define lb lower_bound 
#define ub upper_bound 

int mod=1e9+7;
const int mxn=301000;

void solve(){
    int n,k; cin>>n>>k;
    int a[n]; f(i,0,n) cin>>a[i];
    multiset<int> left,right;
    f(i,0,n){
        if(i>=k){
            auto it = left.find(-a[i-k]);
            if(it!=left.end())
                left.erase(it);
            else {
                it = right.find(a[i-k]);
                if (it!=right.end())
                    right.erase(it);
            }
        }
        if(-*left.begin()<=a[i]) right.insert(a[i]);
        else left.insert(-a[i]);  
        while(!((sz(left)-sz(right))<=1 && sz(left)>=sz(right))){
            if(sz(right)>sz(left)){
                int val = -*right.begin();
                left.insert(val);
                right.erase(right.find(-val));
            }else{
                int val = -*left.begin();
                right.insert(val);
                left.erase(left.find(-val));
            }
        }
        if(i>=(k-1)) cout<<-*left.begin()<<" ";
    }
    cout<<"\n";
}
 
int32_t main(){
    ios_base::sync_with_stdio(false);cin.tie(NULL); 
    srand(time(0));     
    #ifndef ONLINE_JUDGE
        freopen("input.txt","r",stdin);
        freopen("output.txt","w",stdout);
    #endif
    int t=1; //cin>>t;
    while(t--) solve();
    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Problem : Sliding median Cost, median cost to bring each subarray of size k to same value.

// focus this moment !!
 
#include<bits/stdc++.h> 
#include <iostream> 
#include <vector> 
#include <iterator>
using namespace std;
#define int long long
#define ld long double
#define pb push_back
#define ff first
#define ss second
#define f(i,x,n) for(int i=x;i<(int)n;++i)
#define vi vector<int>
#define vvi vector<vector<int>>
#define vvvi vector<vector<vector<int>>>
#define pq priority_queue<int>
#define pqs priority_queue<int,vi,greater<int>> 
#define vpii vector<pair<int,int>>
#define pii pair<int,int>
#define all(x) x.begin(),x.end()
#define sz(x) (int)x.size()
#define mpi map<int,int>
#define lb lower_bound 
#define ub upper_bound 

int mod=1e9+7;
const int mxn=301000;

void solve(){
    int n,k; cin>>n>>k;
    int a[n]; f(i,0,n) cin>>a[i];
    multiset<int> left,right;
    int left_sum=0 ,right_sum = 0;
    f(i,0,n){
        if(i>=k){
            auto it = left.find(-a[i-k]);
            if(it!=left.end()){
                left_sum -= -*it;
                left.erase(it);
            }
            else {
                it = right.find(a[i-k]);
                if (it!=right.end()){
                    right.erase(it);
                    right_sum -= *it;
                }
            }
        }
        if(-*left.begin()<=a[i]) { right.insert(a[i]); right_sum += a[i]; }
        else { left.insert(-a[i]);  left_sum += a[i]; }
        while(!((sz(left)-sz(right))<=1 && sz(left)>=sz(right))){
            if(sz(right)>sz(left)){
                int val = -*right.begin();
                left.insert(val);
                left_sum -= val;
                right.erase(right.find(-val));
                right_sum += val;
            }else{
                int val = -*left.begin();
                right.insert(val);
                right_sum += val;
                left.erase(left.find(-val));
                left_sum -= val;
            }
        }
        if(i>=(k-1)) {
            int med = -*left.begin();
            cout<< (sz(left)*med) - left_sum + right_sum - (sz(right)*med) << " ";
        }
    }
    cout<<"\n";
}
 
int32_t main(){
    ios_base::sync_with_stdio(false);cin.tie(NULL); 
    srand(time(0));     
    #ifndef ONLINE_JUDGE
        freopen("input.txt","r",stdin);
        freopen("output.txt","w",stdout);
    #endif
    int t=1; //cin>>t;
    while(t--) solve();
    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Problem :  Longest Spanning Subset

#include <bits/stdc++.h>
#include <sys/resource.h>
using namespace std;
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
#define ll long long
#define ff first
#define ss second
#define INF 1000000000000000000
#define pb push_back
#define vl vector<ll>
#define pll pair<ll,ll>
#define vll vector<pair<ll,ll>>
#define vi vector<int>
#define sz(a) (int)a.size()
#define all(v) v.begin(), v.end()
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
#define PI 3.141592653589793
#define ldb long double
#define o_set tree<ll, null_type, less<ll>, rb_tree_tag, tree_order_statistics_node_update>
#define o_setpll tree<pair<ll,ll>, null_type, less<pair<ll,ll>>, rb_tree_tag, tree_order_statistics_node_update>
//member functions :
//1. order_of_key(k) : number of elements strictly lesser than k
//2. find_by_order(k) : k-th element in the set
int getRand(int l, int r)
{
    uniform_int_distribution<int> uid(l, r);
    return uid(rng);
}
ll power(ll b, ll e, ll m) {
    if (e == 0) return 1;
    if (e & 1) return b * power(b * b % m, e / 2, m) % m;
    return power(b * b % m, e / 2, m);
}
ll power( ll b, ll e)
{
    if (e == 0) return 1;
    if (e & 1) return b * power(b * b, e / 2);
    return power(b * b, e / 2);
}
ll modI(ll a, ll m)
{
    ll m0 = m, y = 0, x = 1;
    if (m == 1)return 0;
    while (a > 1) {
        ll q = a / m;
        ll t = m; m = a % m;
        a = t; t = y;
        y = x - q * y; x = t;
    }
    if (x < 0) {x += m0;}
    return x;
}
ll lcm(ll a, ll b) {
    return (a / __gcd(a, b)) * b;
}
// struct Edge {
//     int u, v, weight;
//     bool operator<(Edge const& other) {
//         return weight < other.weight;
//     }
// };
vector<int> sort_cyclic_shifts(string const& s) {
    int n = s.size();
    const int alphabet = 256;
    vector<int> p(n), c(n), cnt(max(alphabet, n), 0);
    for (int i = 0; i < n; i++)
        cnt[s[i]]++;
    for (int i = 1; i < alphabet; i++)
        cnt[i] += cnt[i - 1];
    for (int i = 0; i < n; i++)
        p[--cnt[s[i]]] = i;
    c[p[0]] = 0;
    int classes = 1;
    for (int i = 1; i < n; i++) {
        if (s[p[i]] != s[p[i - 1]])
            classes++;
        c[p[i]] = classes - 1;
    }
    vector<int> pn(n), cn(n);
    for (int h = 0; (1 << h) < n; ++h) {
        for (int i = 0; i < n; i++) {
            pn[i] = p[i] - (1 << h);
            if (pn[i] < 0)
                pn[i] += n;
        }
        fill(cnt.begin(), cnt.begin() + classes, 0);
        for (int i = 0; i < n; i++)
            cnt[c[pn[i]]]++;
        for (int i = 1; i < classes; i++)
            cnt[i] += cnt[i - 1];
        for (int i = n - 1; i >= 0; i--)
            p[--cnt[c[pn[i]]]] = pn[i];
        cn[p[0]] = 0;
        classes = 1;
        for (int i = 1; i < n; i++) {
            pair<int, int> cur = {c[p[i]], c[(p[i] + (1 << h)) % n]};
            pair<int, int> prev = {c[p[i - 1]], c[(p[i - 1] + (1 << h)) % n]};
            if (cur != prev)
                ++classes;
            cn[p[i]] = classes - 1;
        }
        c.swap(cn);
    }
    return p;
}

vector<int> build_suffix_array(string s) {
    // s += "$";
    vector<int> sorted_shifts = sort_cyclic_shifts(s);
    sorted_shifts.erase(sorted_shifts.begin());
    return sorted_shifts;
}
vector<int> lcp_construction(string const& s, vector<int> const& p) {
    int n = s.size();
    vector<int> rank(n, 0);
    for (int i = 0; i < n; i++)
        rank[p[i]] = i;

    int k = 0;
    vector<int> lcp(n, 0);
    for (int i = 0; i < n; i++, k ? k-- : 0) {
        if (rank[i] == n - 1) {
            k = 0;
            continue;
        }
        int j = p[rank[i] + 1];
        while (i + k < n && j + k < n && s[i + k] == s[j + k])
            k++;
        lcp[rank[i]] = k;
        if (k)
            k--;
    }
    return lcp;
}

const int N = 101;
int parent[N];
int siz[N];
void make_set(int v) {
    parent[v] = v;
    siz[v] = 1;
}

int find_set(int v) {
    if (v == parent[v])
        return v;
    return find_set(parent[v]);
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (siz[a] < siz[b])
            swap(a, b);
        parent[b] = a;
        siz[a] += siz[b];
    }
}
void solve() {
    int n;
    cin >> n;
    vector<string>v(n);
    string s = "";
    vl indices;
    vl khatam(n);
    for (int i = 0; i < n; i++) {
        cin >> v[i];
        for (int z = 0; z < sz(v[i]) ; z++) {
            indices.pb(i);
        }
        s += v[i];
        khatam[i] = sz(s) - 1;
        indices.pb(-1);
        s += '$';
    }

    vector<int>suf, lcp;
    suf = build_suffix_array(s);
    lcp = lcp_construction(s, suf);
    // cout << sz(suf) << " ";
    ll cost = 0;
    map<pair<int, int>, int>mm;
    for (int i = 0; i < sz(suf) - 1; i++) {
        if (indices[suf[i]] < 0 or indices[suf[i + 1]] < 0) {continue;}
        if (indices[suf[i]] != indices[suf[i + 1]] and lcp[i] > 0) {
            // cout << suf[i] << " " << suf[i + 1] << " -> ";
            // cout << s.substr(suf[i], lcp[i + 1]) << " " << s.substr(suf[i + 1], lcp[i + 1]) << "\n";
            ll a = indices[suf[i]], b = indices[suf[i + 1]];

            // auto it = upper_bound(all(indices), a);
            // it--;
            int len = khatam[a] - suf[i] + 1;
            lcp[i] = min(lcp[i], len); // kitna max length ho skta h ?
            len = khatam[b] - suf[i + 1] + 1;
            lcp[i] = min(lcp[i], len); // kitna max length ho skta h ?


            if (a > b) {swap(a, b);}
            // cost = (cost - mm[ {a, b}] + max(lcp[i + 1], mm[ {a, b}]));
            mm[ {a, b}] = max(lcp[i], mm[ {a, b}]);
        }
    }

    vector<pair<ll, pair<ll, ll>>> edges;
    for (auto x : mm) {
        // cout << x.ff.ff << " " << x.ff.ss << " -> " << x.ss << "\n";
        edges.pb({x.ss, x.ff});
    }

    sort(all(edges));
    reverse(all(edges));
    for (int i = 0; i < n; i++) {make_set(i);}
    for (auto x : edges) {
        if (find_set(x.ss.ff)  != find_set(x.ss.ss)) {
            union_sets(x.ss.ff, x.ss.ss);
            cost += x.ff;
        }
    }

    cout << cost << "\n";
}

int main() {
    ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
    rlimit R;
    getrlimit(RLIMIT_STACK, &R);
    R.rlim_cur = R.rlim_max;
    setrlimit(RLIMIT_STACK, &R);
// freopen("test_input.txt", "r", stdin);
// freopen("test_output.txt", "w", stdout);
    int t = 1;
    cin >> t;
    for (int ntc = 1; ntc <= t; ntc++) {
        // cout << "Case #" << ntc << ": ";
        solve();
    }
    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Problems : Leetcode 256

class Solution {
public:
    int minimumDifference(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());
        int ret = nums[k - 1] - nums[0];
        for (int i = 0; i + k - 1 < nums.size(); i++) {
            ret = min(ret, nums[i + k - 1] - nums[i]);
        }
        return ret;
    }
};

class Solution {
public:
    string kthLargestNumber(vector<string>& nums, int k) {
        vector<pair<int, string>> vs;
        for (auto s : nums) vs.emplace_back(s.size(), s);
        sort(vs.rbegin(), vs.rend());
        return vs[k - 1].second;
    }
};

class Solution {
public:
    int minSessions(vector<int>& tasks, int sessionTime) {
        const int N = tasks.size();
        vector<int> dp(1 << N, 1000);
        dp[0] = 0;
        for (int s = 1; s < 1 << N; s++) {
            int req = 0;
            for (int i = 0; i < N; i++) if ((s >> i) & 1) req += tasks[i];
            if (req <= sessionTime) dp[s] = 1;
            for (int t = s; ; t = (t - 1) & s) {
                if (dp[s] > dp[t] + dp[s - t]) dp[s] = dp[t] + dp[s - t];
                if (t == 0) break;
            }
        }
        return dp.back();
    }
};

class Solution {
public:
    int numberOfUniqueGoodSubsequences(string binary) {
        constexpr int md = 1000000007;
        vector<int> dp{0, 1};
        long long ret = 0;
        for (auto c : binary) {
            int t = c - '0';
            ret += dp[t];
            (dp[t ^ 1] += dp[t]) %= md;
        }
        if (binary != string(binary.size(), '1')) ret++;
        return ret % md;
    }
};

// -----------------------------------------------------------------------------------------------------------------------------------

// Deltix Round Summer 2021

// Problem : B

cin>>t;
while(t--) {
    cin>>n;
    a.resize(n);
    vector<int> s[2];
    for(int i=0;i<n;++i) {
        cin>>a[i];a[i]&=1;
        s[a[i]].pb(i);
    }
    ll aa=2e18;
    for(int o=0;o<2;++o) {
        int c[2]={0,0};
        ll ans=0;
        for(int i=0;i<n;++i) {
            int g=(o+i)&1;++c[g];
            if(g==0&&c[g]<=s[0].size()) ans+=abs(i-s[0][c[g]-1]);
        }
        if(c[0]==s[0].size()) aa=min(aa,ans);
    }
    if(aa>1e18) aa=-1;
    cout<<aa<<'\n';
}

// Problem : A

int main() {
    int t;
    cin >> t;
    for (int i = 1; i <= t; i++) {
        int c, d;
        scanf("%d%d", &c, &d);
        int ans = -1;
        if ((c % 2) != (d % 2)) ans = -1;
        else if (c == d) ans = 1;
        else ans = 2;
        if (c == 0 && d == 0) ans = 0;
        printf("%d\n", ans);
    }
    return (0-0); //<3
}


// Problem: D

void ff()
{
    fflush(stdout);
}
 
int n, k;
 
int ukr[]={1,6,4,2,3,5,4};
 
int pytand(int a, int b)
{
    printf("and %d %d\n", a, b);
    ff();
    int x;
    scanf("%d", &x);
    //~ x=ukr[a-1]&ukr[b-1];
    return x;
}
 
int pytor(int a, int b)
{
    printf("or %d %d\n", a, b);
    ff();
    int x;
    scanf("%d", &x);
    //~ x=ukr[a-1]|ukr[b-1];
    return x;
}
 
ll suma(int a, int b)
{
    return pytor(a, b)+pytand(a, b);
}
 
vll known;
 
int main()
{
    scanf("%d%d", &n, &k);
    ll a=suma(1, 2);
    ll b=suma(1, 3);
    ll c=suma(2, 3);
    known.push_back(a+b-c);
    known.push_back(a+c-b);
    known.push_back(b+c-a);
    for (ll &i : known)
        i/=2;
    ll odj=known[0];
    for (int i=4; i<=n; i++)
        known.push_back(suma(1, i)-odj);
    //~ debug() << known;
    sort(known.begin(), known.end());
    printf("finish %lld\n", known[k-1]);
    ff();
    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// minimum segment tree

vector<int> vt;

void st(int curr,int l,int r,int id,int val){
    if(l==r){
        if(l==id) vt[curr]=val;
        return ;
    }
    if(r<id || id<l) return ;
    int mid=l+r; mid/=2;
    st(2*curr+1,l,mid,id,val);
    st(2*curr+2,mid+1,r,id,val);
    vt[curr]=min(vt[2*curr+1],vt[2*curr+2]);
}
 
int gt(int curr,int l,int r,int L,int R){
    if(l==r){
        if(l>=L && l<=R) return vt[curr];
        return INT_MAX;
    }
    if(R<l || r<L) return INT_MAX;
    if(l>=L && r<=R) return vt[curr];
    int mid=l+r; mid/=2;
    return min(gt(2*curr+1,l,mid,L,R), gt(2*curr+2,mid+1,r,L,R));
}

// -----------------------------------------------------------------------------------------------------------------------------------

/*
Problem Name: Path Queries II
Problem Link: https://cses.fi/problemset/task/2134
Author: Sachin Srivastava (mrsac7)
*/
#include "bits/stdc++.h"
using namespace std;

#define int long long
#define endl '\n'

const int mxN = 2e5+5;
vector<int> adj[mxN];
int dp[mxN], depth[mxN], par[mxN];
int heavy[mxN], head[mxN], id[mxN];
int seg[10*mxN];
int N;
int val[mxN];

void update(int k, int x) {
    k += N; seg[k] = x; k >>= 1;
    while (k > 0) {
        seg[k] = max(seg[2*k], seg[2*k+1]);
        k >>= 1;
    }
}

int query(int a, int b) {
    a += N, b += N;
    int s = 0;
    while (a <= b) {
        if (a & 1) {
            s = max(s, seg[a]);
            a++;
        }
        if (~b & 1) {
            s = max(s, seg[b]);
            b--;
        }
        a >>= 1, b >>= 1;
    }
    return s;
}

void dfs(int s, int p) {
    dp[s] = 1;
    int mx = 0;
    for (auto i: adj[s]) if (i != p) {
        par[i] = s;
        depth[i] = depth[s] + 1;
        dfs(i, s);
        dp[s] += dp[i];        
        if (dp[i] > mx)
            mx = dp[i], heavy[s] = i;
    }
}

int cnt = 0;
void hld(int s, int h) {
    head[s] = h;
    id[s] = ++cnt;
    update(id[s]-1, val[s]);
    if (heavy[s])
        hld(heavy[s], h);
    for (auto i: adj[s]) {
        if (i != par[s] && i != heavy[s])
            hld(i, i);
    }
}

int path(int x, int y){
    int ans = 0;
    while (head[x] != head[y]) {
        if (depth[head[x]] > depth[head[y]])
            swap(x, y);
        ans = max(ans, query(id[head[y]]-1, id[y]-1));
        y = par[head[y]];
    }
    if(depth[x] > depth[y]) 
        swap(x, y);
    ans = max(ans, query(id[x]-1, id[y]-1));
    return ans;
}


signed main(){
    ios_base::sync_with_stdio(false);cin.tie(0);cout.tie(0);
    #ifdef LOCAL
    freopen("input.txt", "r" , stdin);
    freopen("output.txt", "w", stdout);
    #endif
    
    int n, t; cin>>n>>t;
    N = 1 << (int) ceil(log2(n));
    for (int i = 1; i <= n; i++) 
        cin>>val[i];

    for (int i = 1; i < n; i++) {
        int x, y; cin>>x>>y;
        adj[x].push_back(y);
        adj[y].push_back(x);
    }
    dfs(1, 0);
    hld(1, 1);

    while (t--) {
        int ch; cin>>ch;
        if (ch == 1) {
            int k, x; cin>>k>>x;
            update(id[k]-1, x);
        }
        else {
            int x, y; cin>>x>>y;
            cout << path(x, y) << ' ';
        }
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Fixed Length Paths I CSES

#include<bits/stdc++.h>
using namespace std;
 
int main() {
    using i64 = int64_t;
    cout.tie(nullptr)->sync_with_stdio(false);
    int N, K; cin >> N >> K;
 
    vector<vector<int>> adj(N);
    for (int i = 0; i + 1 < N; ++i) {
        int a, b; cin >> a >> b;
        --a, --b;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }
    
    i64 cnt = 0;
    map<int, i64> large;
    auto dfs = [&](auto self, int v, int p, int h, map<int, i64>& large) -> void {
        large[h]++;
        for (int nxt : adj[v]) {
            if (nxt == p) continue;
            map<int, i64> small;
            self(self, nxt, v, h + 1, small);
            if (small.size() > large.size()) swap(small, large);
            for (auto [a, b] : small) {
                cnt += large[K - a + 2 * h] * b;
            }
            for (auto [a, b] : small) large[a] += b;
        }
    };
    dfs(dfs, 0, -1, 0, large);
    cout << cnt << '\n';
}

// -----------------------------------------------------------------------------------------------------------------------------------


/*
https://cses.fi/problemset/task/2137
Beautiful Subgrids
*/

#include <stdio.h>
 
#define N   3000
#define L   60
#define K   (N / L)
#define B   (1 << 20)
 
char kk[B];
 
void init() {
    int b;
 
    for (b = 1; b < B; b++)
        kk[b] = kk[b & b - 1] + 1;
}
 
char count(long long b) {
    return kk[b >> 40] + kk[b >> 20 & (1 << 20) - 1] + kk[b & (1 << 20) - 1];
}
 
int main() {
    static long long bb[N][K];
    int n, h, i, j;
    long long ans;
 
    init();
    scanf("%d", &n);
    for (i = 0; i < n; i++) {
        static char s[N + 1];
 
        scanf("%s", s);
        for (j = 0; j < n; j++)
            if (s[j] == '1')
                bb[i][j / L] |= 1LL << j % L;
    }
    ans = 0;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++) {
            short k;
 
            k = 0;
            for (h = 0; h < K; h++)
                k += count(bb[i][h] & bb[j][h]);
            ans += k * (k - 1) / 2;
        }
    printf("%lld\n", ans);
    return 0;
}


// -----------------------------------------------------------------------------------------------------------------------------------

// Atcoder 218 : H 
// Author :  ygussany

#include <stdio.h>

typedef struct {
    int key, id;
} data;

typedef struct {
    data obj[200001];
    int size;
} max_heap;

void push(max_heap* h, data x)
{
    int i = ++(h->size), j = i >> 1;
    data tmp;
    h->obj[i] = x;
    while (j > 0) {
        if (h->obj[i].key > h->obj[j].key) {
            tmp = h->obj[j];
            h->obj[j] = h->obj[i];
            h->obj[i] = tmp;
            i = j;
            j >>= 1;
        } else break;
    }
}

data pop(max_heap* h)
{
    int i = 1, j = 2;
    data output = h->obj[1], tmp;
    h->obj[1] = h->obj[(h->size)--];
    while (j <= h->size) {
        if (j < h->size && h->obj[j^1].key > h->obj[j].key) j ^= 1;
        if (h->obj[j].key > h->obj[i].key) {
            tmp = h->obj[j];
            h->obj[j] = h->obj[i];
            h->obj[i] = tmp;
            i = j;
            j <<= 1;
        } else break;
    }
    return output;
}

long long solve(int N, int R, int A[])
{
    int i, next[200001], prev[200001];
    max_heap h;
    data d;
    h.size = 0;
    for (i = 1, A[0] = 0, A[N] = 0; i <= N; i++) {
        next[i-1] = i;
        prev[i] = i - 1;
        d.key = A[i-1] + A[i];
        d.id = i - 1;
        push(&h, d);
    }
    if (R * 2 > N) R = N - R;

    int j;
    long long ans = 0;
    while (h.size > 0 && R > 0) {
        d = pop(&h);
        i = d.id;
        if (next[i] < 0 || d.key != A[i] + A[next[i]]) continue;
        R--;
        ans += d.key;
        if (i > 0 && next[i] < N) {
            next[prev[i]] = next[next[i]];
            prev[next[next[i]]] = prev[i];
            d.key = A[prev[i]] + A[next[prev[i]]];
            d.id = prev[i];
            push(&h, d);
        } 
        next[next[i]] = -1;
        next[i] = -1;
    }
    return ans;
}

int main()
{
    int i, N, R, A[200001];
    scanf("%d %d", &N, &R);
    for (i = 1; i < N; i++) scanf("%d", &(A[i]));
    printf("%lld\n", solve(N, R, A));
    fflush(stdout);
    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Problem : Atcoder 216 F
/* Author: Thallium54 {{{
 * Blog: https://blog.tgc-thallium.com/
 * Code library: https://github.com/thallium/acm-algorithm-template
 *//*}}}*/

#include <bits/stdc++.h>
using namespace std;
#define all(x) (x).begin(),(x).end() //{{{
#ifndef LOCAL // https://github.com/p-ranav/pprint
#define de(...)
#define de2(...)
#endif
using ll = long long;
using pii = pair<int, int>; //}}}
 
template <int MOD>
struct ModInt {
    int val;
    // constructor
    ModInt(ll v = 0) : val(int(v % MOD)) {
        if (val < 0) val += MOD;
    };
    // unary operator
    ModInt operator+() const { return ModInt(val); }
    ModInt operator-() const { return ModInt(MOD - val); }
    ModInt inv() const { return this->pow(MOD - 2); }
    // arithmetic
    ModInt operator+(const ModInt& x) const { return ModInt(*this) += x; }
    ModInt operator-(const ModInt& x) const { return ModInt(*this) -= x; }
    ModInt operator*(const ModInt& x) const { return ModInt(*this) *= x; }
    ModInt operator/(const ModInt& x) const { return ModInt(*this) /= x; }
    ModInt pow(ll n) const {
        auto x = ModInt(1);
        auto b = *this;
        while (n > 0) {
            if (n & 1) x *= b;
            n >>= 1;
            b *= b;
        }
        return x;
    }
    // compound assignment
    ModInt& operator+=(const ModInt& x) {
        if ((val += x.val) >= MOD) val -= MOD;
        return *this;
    }
    ModInt& operator-=(const ModInt& x) {
        if ((val -= x.val) < 0) val += MOD;
        return *this;
    }
    ModInt& operator*=(const ModInt& x) {
        val = int(ll(val) * x.val % MOD);
        return *this;
    }
    ModInt& operator/=(const ModInt& x) { return *this *= x.inv(); }
    // compare
    bool operator==(const ModInt& b) const { return val == b.val; }
    bool operator!=(const ModInt& b) const { return val != b.val; }
    // I/O
    friend std::istream& operator>>(std::istream& is, ModInt& x) noexcept { return is >> x.val; }
    friend std::ostream& operator<<(std::ostream& os, const ModInt& x) noexcept { return os << x.val; }
};
using mint = ModInt<int(998244353)>;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> a(n), b(n);
    for (auto& x : a) cin >> x;
    for (auto& x : b) cin >> x;
 
    int mx = *max_element(begin(a), end(a));
    vector<int> ids(n);
    iota(begin(ids), end(ids), 0);
    sort(begin(ids), end(ids), [&](int i, int j) { return a[i] < a[j]; });
    vector<mint> dp(mx+1);
    dp[0]=1;
    mint ans;
    for (auto i : ids) {
        for (int j = 0; j <= a[i] - b[i]; j++)
            ans+=dp[j];
        for (int j=mx; j>=b[i]; j--)
            dp[j]+=dp[j-b[i]];
    }
    cout << ans << endl;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Codeforces: 743 Book

#include <bits/stdc++.h>

using namespace std;

void solve() {
    int n;
    cin >> n;
    vector<vector<int>> e(n);
    vector<int> deg(n);
    for(int i = 0; i < n; i++) {
        int k;
        cin >> k;
        while(k--) {
            int j;
            cin >> j;
            j--;
            e[j].push_back(i);
            deg[i]++;
        }
    }
    set<int> s;
    for(int i = 0; i < n; i++) {
        if(deg[i] == 0) s.insert(i);
    }
    int ans = 1, last = -1;
    for(int iter = 0; iter < n; iter++) {
        if(s.empty()) {
            ans = -1;
            break;
        }
        auto it = s.lower_bound(last);
        if(it == s.end()) {
            it = s.begin();
            ans++;
        }
        int i = *it;
        s.erase(it);
        for(int j : e[i]) {
            deg[j]--;
            if(!deg[j]) {
                s.insert(j);
            }
        }
        last = i;
    }
    cout << ans << '\n';
}

int main() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
    int t;
    cin >> t;
    while(t--) {
        solve();
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Codeforces: 743  Xor of 3

#include <bits/stdc++.h>

int main() {
    using namespace std;
    ios_base::sync_with_stdio(false), cin.tie(nullptr);

    int T; cin >> T;
    while (T--) {
        int N; cin >> N;
        std::vector<int> A(N);
        for (auto& a : A) cin >> a;

        auto ans = [&]() -> std::optional<std::vector<int>> {
            int tot = 0;
            for (auto a : A) tot ^= a;
            if (tot != 0) {
                return std::nullopt;
            }
            if (N & 1) {
                std::vector<int> resp; resp.reserve(N);
                for (int i = 0; i <= N-3; i += 2) {
                    resp.push_back(i);
                }
                for (int i = N-5; i >= 0; i -= 2) {
                    resp.push_back(i);
                }
                return resp;
            } else {
                int cur_tot = A[0];
                for (int md = 1; md < N; md += 2) {
                    if (cur_tot == 0) {
                        std::vector<int> resp; resp.reserve(N);
                        for (int i = 0; i <= md-3; i += 2) {
                            resp.push_back(i);
                        }
                        for (int i = md-5; i >= 0; i -= 2) {
                            resp.push_back(i);
                        }
                        for (int i = md; i <= N-3; i += 2) {
                            resp.push_back(i);
                        }
                        for (int i = N-5; i >= md; i -= 2) {
                            resp.push_back(i);
                        }
                        return resp;
                    }
                    if (md + 2 < N) {
                        cur_tot ^= A[md] ^ A[md+1];
                    }
                }
                return std::nullopt;
            }
        }();

        if (ans) {
            cout << "YES" << '\n';
            cout << ans->size() << '\n';
            for (int z = 0; z < int(ans->size()); z++) {
                cout << (*ans)[z]+1 << " \n"[z+1==int(ans->size())];
            }
        } else {
            cout << "NO" << '\n';
        }
    }

    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Problem : Subsequence LCS 
// Codechef Lunchtime sept 2021

// elegant way of finding longest increasing subsequence in n^2
vector<bool> findLis(vector<ll> v,ll n) {
    vector< bool> fg(n,false);
    vector< ll > dp(n,1);
    ll ans=0;
    for(int i=1;i<n;i++) {
        for(int j=0;j<i;j++)
            if(v[i]>v[j])
                dp[i] = max(dp[i],dp[j]+1);
        ans = max(dp[i],ans);
    }
    for(ll i=n-1;i>=0;i--) {
        if(dp[i]==ans) {
            fg[i] = true;
            ans--;
        }
    }
    return fg;
}


/**
 🍪 the_hyp0cr1t3
 🍪 25.09.2021 19:36:08
**/
#ifdef W
    #include <k_II.h>
#else
    #include <bits/stdc++.h>
    using namespace std;
#endif

template<class T, bool ONE = true>
class Segtree {
    int N; vector<T> st;

    static int ceil2n(int x) {
        return (1 << 31 - __builtin_clz(x << 1) + !!(x & x-1)) + 2;
    }

    template<class It>
    void build(int node, int L, int R, It beg, It end, bool opt) {
        if(L == R) return st[node] = static_cast<T>(*beg), void();
        int M = L + R >> 1;
        build(node << 1, L, M, beg, beg + (opt? M - L : 0), opt);
        build(node << 1 | 1, M + 1, R, beg + (opt? M - L + 1 : 0), end, opt);
        st[node] = T(st[node << 1], st[node << 1 | 1]);
    }

    T Query(int node, int L, int R, int ql, int qr) {
        if(ql == L and R == qr) return st[node];
        int M = L + R >> 1;
        return qr <= M? Query(node << 1, L, M, ql, qr)
                        : M < ql? Query(node << 1 | 1, M + 1, R, ql, qr)
                            : T(Query(node << 1, L, M, ql, M),
                                    Query(node << 1 | 1, M + 1, R, M + 1, qr));
    }

    void Update(int node, int L, int R, int pos, int64_t val) {
        if(L == R) return st[node].upd(val);
        int M = L + R >> 1;
        pos <= M? Update(node << 1, L, M, pos, val)
                    : Update(node << 1 | 1, M + 1, R, pos, val);
        st[node] = T(st[node << 1], st[node << 1 | 1]);
    }

public:

    template<class... Args>
    Segtree(int N, Args&&... args): N(N), st(ceil2n(N)) {
        T val(forward<Args>(args)...); build(1, 1, N - !ONE, &val, &val, false);
    }

    template<typename It,
        typename = enable_if_t<is_same<typename iterator_traits<It>::iterator_category,
                                            random_access_iterator_tag>::value,
                                typename iterator_traits<It>::difference_type>>
    Segtree(It beg, It end): N(end - beg), st(ceil2n(N)) {
        build(1, ONE, N - !ONE, beg, end, true);
    }

    T query(int pos) { return Query(1, ONE, N - !ONE, pos, pos); }
    T query(int l, int r) { return Query(1, ONE, N - !ONE, l, r); }
    void update(int pos, int64_t val) { Update(1, ONE, N - !ONE, pos, val); }

};

struct Node {
    int val;
    Node(int val = 0): val(val) {}
    Node(const Node& l, const Node& r)
        : val(max(l.val, r.val)) {}
    void upd(int delta) { val = max(val, delta); }
    operator int() const { return val; }
};

int main() {
    ios_base::sync_with_stdio(false), cin.tie(nullptr);
    int Q; cin >> Q;

    while(Q--) []() {
        int i, j, n, mxi, mxd;
        cin >> n;
        vector<int> a(n);
        for(auto& x: a) cin >> x;
        a.insert(a.begin(), -1);

        vector<int> incl(n + 1), incr(n + 1);
        vector<int> decl(n + 1), decr(n + 1);

        {
            Segtree<Node> inc(n), dec(n);
            for(i = 1; i <= n; i++) {
                incl[i] = inc.query(1, a[i]);
                decl[i] = dec.query(a[i], n);

                inc.update(a[i], incl[i] + 1);
                dec.update(a[i], decl[i] + 1);
            }

            mxi = inc.query(1, n);
            mxd = dec.query(1, n);
        }

        {
            Segtree<Node> inc(n), dec(n);
            for(i = n; i; i--) {
                incr[i] = inc.query(a[i], n);
                decr[i] = dec.query(1, a[i]);

                inc.update(a[i], incr[i] + 1);
                dec.update(a[i], decr[i] + 1);
            }
        }

        vector<vector<int>> pos(n + 1);
        for(i = 1; i <= n; i++)
            pos[a[i]].push_back(i);

        int ans = 0;
        for(int v = 1; v <= n; v++) {  // there can be only one of common integer in both sequence.
            int ansi = 0, ansd = 0;
            for(i = j = 0; i < pos[v].size(); i++) { // iterate over all staring position since you do not know them , 1 1 2 1 1 2 2. 
                for(j = max(j, i); j < pos[v].size(); j++) {   // this is essentially like two pointers.
                    int cnti = incl[pos[v][i]] + j - i + 1 + incr[pos[v][j]];
                    if(cnti < mxi) break;
                    ansi = max(ansi, j - i + 1);
                }
            }
            for(i = j = 0; i < pos[v].size(); i++) {
                for(j = max(j, i); j < pos[v].size(); j++) {
                    int cntd = decl[pos[v][i]] + j - i + 1 + decr[pos[v][j]];
                    if(cntd < mxd) break;
                    ansd = max(ansd, j - i + 1);
                }
            }
            ans = max(ans, min(ansi, ansd)); // answer can be updated with the minimum number of most of common integers in both sequence.
        }

        cout << ans << '\n';
    }();

} // ~W

// -----------------------------------------------------------------------------------------------------------------------------------

// Problem G: codeforces 744. author : sikdangcopo 
#include <bits/stdc++.h>
#define gibon ios::sync_with_stdio(false); cin.tie(0);
#define bp __builtin_popcount
#define fir first
#define sec second
#define pii pair<int, int>
#define pll pair<ll, ll>
#define pmax pair<ll, ll>
#pragma GCC optimize("O3")
#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
typedef long long ll;
using namespace std;
int dx[4]={0, 1, 0, -1}, dy[4]={1, 0, -1 , 0};
const int mxN=10020;
const int mxM=300000;
const int mxK=40;
const int MOD=1000000007;
const ll P1=1000000007, P2=10000000009;
const ll INF=8000000000000000001;
int N;
int A[mxN];
bool Chk[mxN][2];
bool solv(int tval)
{
    for(int i=0;i<=tval;i++)    Chk[i][1]=true;
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<=tval;j++)    Chk[j][0]=Chk[j][1], Chk[j][1]=false;
        for(int j=0;j<=tval;j++)    if(Chk[j][0])
        {
            if(j+A[i]<=tval)    Chk[j+A[i]][1]=true;
            if(j-A[i]>=0)   Chk[j-A[i]][1]=true;
        }
    }
    for(int i=0;i<=tval;i++)    if(Chk[i][1])  return true;
    return false;
}
int main()
{
    gibon
    int T;
    cin >> T;
    while(T--)
    {
        cin >> N;
        for(int i=0;i<N;i++)    cin >> A[i];
        int s=0, e=2000;
        while(s!=e)
        {
            int mid=(s+e)/2;
            if(solv(mid))   e=mid;
            else    s=mid+1;
        }
        cout << e << '\n';
    }
}

// -----------------------------------------------------------------------------------------------------------------------------------

// Sum of all subarray xor.

#include <bits/stdc++.h>
using namespace std;
#define ll long long
int main() {
    // your code goes here
    int t; cin>>t;
    while(t--){
        ll n; cin>>n;
        vector<ll> a(n+1, 0);
        vector<ll> o(32, 0), z(32, 0);
        for(int j=0; j<32; j++) z[j]++;
        for(int i=0; i<n; i++) {
            cin>>a[i+1];
            a[i+1]^=a[i];
            for(int j=0; j<32; j++){
                if(a[i+1]&(1<<j)){
                    o[j]++;                
                }
                else z[j]++;
            }
        }
        ll ans = 0;
        //for(int j=0; j<32; j++) cout<<z[j]<<" ";
        for(int i=0; i<n+1; i++){
            for(int j=0; j<32; j++){
                if(a[i]&(1<<j)){
                    ans += z[j]*(1<<j);
                }
                else{
                    ans += o[j]*(1<<j);
                }
            }
        }
        cout<<(ans>>1LL)<<"\n";
    }
    return 0;
}

// -----------------------------------------------------------------------------------------------------------------------------------
