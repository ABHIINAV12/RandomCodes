// Kotlin Practice 7
import java.lang.AssertionError
 
private fun readLn() = readLine()!! // string line
private fun readInt() = readLn().toInt() // single int
private fun readLong() = readLn().toLong() // single long
private fun readDouble() = readLn().toDouble() // single double
private fun readStrings() = readLn().split(" ") // list of strings
private fun readInts() = readStrings().map { it.toInt() } // list of ints
private fun readLongs() = readStrings().map { it.toLong() } // list of longs
private fun readDoubles() = readStrings().map { it.toDouble() } // list of doubles
 
private fun myAssert(x: Boolean) {
    if (!x) {
        throw AssertionError()
    }
}
 
fun main(args: Array<String>) {
    var t = readInt()
    for (i in 0 until t) {
        var (a, b) = readInts()
        println(a + b)
    }
}
 
fun main(args: Array<String>) {
    var t = readInt()
    for (i in 0 until t) {
        var (a, b, k) = readInts()
        println(((k + 1) / 2L) * a - (k / 2L) * b)
    }
}


fun main(args: Array<String>) {
    var t = readInt()
    for (i in 0 until t) {
        var (n, k) = readInts()
        var s = CharArray(n)
        for (i in 0 until n) {
            s[i] = (97 + i % k).toChar()
        }
        println(s.joinToString(""))
    }
}

fun main(args: Array<String>) {
    var n = readInt()
    var a = readInts().sorted()
    var ans = 0
    for (i in 0 until n step 2) {
        ans += a[i + 1] - a[i]
    }
    println(ans)
}


fun main(args: Array<String>) {
    var n = readInt()
    var a = readInts()
    var prefMax = IntArray(n + 1)
    var prefSum = LongArray(n + 1)
    var sufMax = IntArray(n + 1)
    var sufSum = LongArray(n + 1)
    for (i in 0 until n) {
        prefMax[i + 1] = maxOf(prefMax[i], a[i])
        prefSum[i + 1] = prefSum[i] + a[i]
    }
    for (i in n - 1 downTo 0) {
        sufMax[i] = maxOf(sufMax[i + 1], a[i])
        sufSum[i] = sufSum[i + 1] + a[i]
    }
    var nice = ArrayList<Int>()
    for (i in 0 until n) {
        var max = maxOf(prefMax[i], sufMax[i + 1])
        var sum = prefSum[i] + sufSum[i + 1]
        if (2L * max == sum) {
            nice.add(i + 1)
        }
    }
    println(nice.size)
    println(nice.joinToString(" "))
}


fun main(args: Array<String>) {
    var n = readInt()
    var s = Array<String>(2 * n - 2) {""}
    var x = ""
    var y = ""
    for (i in 0 until 2 * n - 2) {
        s[i] = readLn()
        if (s[i].length == n - 1) {
            if (x == "") {
                x = s[i]
            } else {
                y = s[i]
            }
        }
    }
    fun test(w: String) {
        var res = CharArray(2 * n - 2)
        var used = BooleanArray(n)
        for (i in 0 until 2 * n - 2) {
            if (w.startsWith(s[i]) && !used[s[i].length]) {
                res[i] = 'P'
                used[s[i].length] = true
                continue
            }
            if (w.endsWith(s[i])) {
                res[i] = 'S'
                continue
            }
            return
        }
        println(res.joinToString(""))
        exitProcess(0)
    }
    test(x[0] + y)
    test(y[0] + x)
    myAssert(false)
}

// ------------------------------------------------------------------------------------------------------

// Kotlin Actual 7

fun main(args: Array<String>) {
    var t = readInt()
    for (i in 0 until t) {
        var (n, k) = readInts()
        var ans = 0
        for (i in 0 until n) {
            var (l, r) = readInts()
            if (k in l..r) {
                ans = maxOf(ans, r - k + 1)
            }
        }
        println(ans)
    }
}


fun main(args: Array<String>) {
    var t = readInt()
    for (i in 0 until t) {
        var n = readInt()
        var a = readInts()
        var fail = false
        for (i in 0 until n - 1) {
            if (a[i] % 2 == a[i + 1] % 2) {
                fail = true
            }
        }
        println(if (fail) "YES" else "NO")
    }
}

fun main(args: Array<String>) {
    var t = readInt()
    for (qq in 0 until t) {
        var (n, k) = readInts()
        var s = readLn()
        var cc = s.count {id -> id == '1'}
        var a = IntArray(n) {1}
        var ptr = 0
        var ans = 0
        while (cc > 0) {
            ans += 1
            a[ptr] = 0
            if (s[ptr] == '1') {
                cc -= 1
                if (cc == 0) {
                    break
                }
            }
            var take = k
            while (take > 0) {
                while (true) {
                    ptr += 1
                    if (ptr == n) {
                        ptr = 0
                    }
                    if (a[ptr] == 1) {
                        break
                    }
                }
                take -= 1
            }
        }
        println(ans)
    }
}


fun main(args: Array<String>) {
    var (n, m) = readInts()
    var s = Array<String>(n) {""}
    for (i in 0 until n) {
        s[i] = readLn()
    }
    s.sort()
    var t = readInt()
    for (i in 0 until t) {
        var w = readLn()
        var ans = 0
        for (j in 0 until m + 1) {
            if (j > 0 && w[j] == w[j - 1]) {
                continue
            }
            var z = w.substring(0, j) + w.substring(j + 1)
            if (s.binarySearch(z) >= 0) {
                ans += 1
            }
        }
        println(ans)
    }
}

fun main(args: Array<String>) {
    var n = readInt()
    var a = readInts().toMutableList()
    var b = readInts().toMutableList()
    a.sort()
    b.sort()
    val inf = 1.01e9.toInt()
    var pref = IntArray(n + 1) {-inf}
    for (i in 0 until n) {
        pref[i + 1] = maxOf(pref[i], b[i] - a[i])
    }
    var suf = IntArray(n + 1) {-inf}
    for (i in n - 1 downTo 0) {
        suf[i] = maxOf(suf[i + 1], b[i + 1] - a[i])
    }
    var m = readInt()
    var c = readInts()
    var res = IntArray(m)
    for (i in 0 until m) {
        var pos = a.binarySearch(c[i])
        if (pos < 0) {
            pos = -pos - 1
        }
        res[i] = maxOf(maxOf(pref[pos], suf[pos]), b[pos] - c[i])
    }
    println(res.joinToString(" "))
}

fun main(args: Array<String>) {
    var s = readLn()
    var n = s.length
    var pref = IntArray(n + 1)
    for (i in 0 until n) {
        pref[i + 1] = pref[i] + (if (s[i] == '1') 1 else 0)
    }
    var res = IntArray(n)
    for (k in 1..n) {
        var pos = 0
        while (pos < n) {
            res[k - 1] += 1
            var low = pos
            var high = n
            while (low < high) {
                var mid = (low + high + 1) shr 1
                var len = mid - pos
                var c1 = pref[mid] - pref[pos]
                var c0 = len - c1
                if (c0 <= k || c1 <= k) {
                    low = mid
                } else {
                    high = mid - 1
                }
            }
            pos = low
        }
    }
    println(res.joinToString(" "))
}

fun main(args: Array<String>) {
    var t = readInt()
    for (qq in 0 until t) {
        var (h, w) = readInts()
        var a = Array(h) {IntArray(w)}
        for (i in 0 until h) {
            a[i] = readInts().toIntArray()
        }
        var cc = 0
        var b = Array(0) {IntArray(0)}
        var seq = ArrayList<Int>()
        fun dfs(i: Int, j: Int) {
            var v = a[i][j]
            a[i][j] = 0
            seq.add(v)
            for (di in -1..1) {
                for (dj in -1..1) {
                    if (di * di + dj * dj == 1) {
                        var ni = i + di
                        var nj = j + dj
                        if (ni >= 0 && nj >= 0 && ni < h && nj < w) {
                            if (a[ni][nj] > 0) {
                                dfs(ni, nj)
                                seq.add(v)
                            }
                        }
                    }
                }
            }
        }
        for (i in 0 until h) {
            for (j in 0 until w) {
                if (a[i][j] > 0) {
                    cc += 1
                    if (cc > 1) {
                        continue
                    }
                    dfs(i, j)
                }
            }
        }
        if (cc > 1) {
            println(-1)
        } else {
            println("${seq.size / 2 + 1} ${seq.size / 2 + 1}")
            for (i in 0 until seq.size / 2 + 1) {
                println(seq.subList(i, i + seq.size / 2 + 1).joinToString(" "))
            }
        }
    }
}

fun main(args: Array<String>) {
    var (h, w) = readInts()
    var s = Array<String>(h) {""}
    for (i in 0 until h) {
        s[i] = readLn()
    }
    var cnt = LongArray(32)
    var st = IntArray(w + 1)
    var ans = LongArray(5)
    for (mask in 1 until 32) {
        var b = IntArray(w)
        for (i in 0 until h) {
            var sum = 0L
            var sz = 1
            st[0] = -1
            for (j in 0 until w) {
                var c = s[i][j].toInt() - 65
                if ((mask and (1 shl c)) != 0) {
                    b[j] += 1
                } else {
                    b[j] = 0
                }
                while (sz > 1 && b[j] <= b[st[sz - 1]]) {
                    sum -= b[st[sz - 1]] * (st[sz - 1] - st[sz - 2])
                    sz -= 1
                }
                sum += b[j] * (j - st[sz - 1])
                st[sz++] = j
                cnt[mask] += sum
            }
        }
        var old = cnt[mask]
        for (sub in 1 until mask) {
            if ((mask and sub) == sub) {
                cnt[mask] -= cnt[sub]
            }
        }
        var pc = 0
        for (i in 0 until 5) {
            pc += (mask shr i) and 1
        }
        ans[pc - 1] += cnt[mask]
    }
    println(ans.joinToString(" "))
}

fun main(args: Array<String>) {
    var (n1, n2, m) = readInts()
    var k = readInts()
    var p = IntArray(n1 + n2) {it}
    var b = BooleanArray(n1 + n2) {true}
    val inf = 1e9.toInt()
    var v = IntArray(n1 + n2) {it -> if (it < n1) k[it] else inf}
    fun get(x: Int): Int {
        if (x != p[x]) {
            p[x] = get(p[x])
        }
        return p[x]
    }
    fun unite(x: Int, y: Int) {
        var px = get(x)
        var py = get(y)
        if (px == py) {
            b[px] = false
        } else {
            p[px] = py
            b[py] = (b[py] and b[px])
            v[py] = minOf(v[py], v[px])
        }
    }
    for (i in 0 until m) {
        var (a, b) = readInts()
        unite(a - 1, n1 + b - 1)
    }
    var ans = 0
    for (i in 0 until n1 + n2) {
        if (get(i) == i && b[i]) {
            ans += v[i]
        }
    }
    println(ans)
}

fun main(args: Array<String>) {
    val M = 500010
    var n = readInt()
    var x = IntArray(n)
    var y = IntArray(n)
    var at = Array(M) {ArrayList<Int>()}
    for (i in 0 until n) {
        var (a, b) = readInts()
        x[i] = a
        y[i] = b
        at[y[i]].add(i)
    }
    var ans = n
    for (left in 0..1) {
        var p = IntArray(n) {it -> if ((y[it] and 1) == left) 1 else 0}
        var st = IntArray(n)
        var aux = IntArray(n)
        var c = ArrayList<Int>()
        for (yy in 0 until M) {
            for (i in at[yy]) {
                c.add(i)
            }
            c.sortWith(compareBy({x[it]}, {p[it]}))
            var cnt = c.size
            for (i in 0 until cnt) {
                aux[i] = c[i]
            }
            c.clear()
            var sz = 0
            for (id in 0 until cnt) {
                var i = aux[id]
                if (p[i] == 1) {
                    st[sz++] = i
                } else {
                    if (sz > 0) {
                        sz -= 1
                        ans -= 1
                    } else {
                        if (y[i] == yy) {
                            c.add(i)
                        }
                    }
                }
            }
            if (sz > 0 && y[st[sz - 1]] == yy) {
                for (i in 0 until sz) {
                    c.add(st[i])
                }
            }
        }
    }
    println(ans)
}

// ------------------------------------------------------------------------------------------------------
