#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 4
#SBATCH -t 00:06:00
#SBATCH -J hrreg
#SBATCH -o /public/home/scnethr2601a10/probe/reg-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/reg-%j.err

R=https://image.ac.com:5000
echo "===== 1. /v2/ challenge ====="
timeout 10 curl -sSk -D- -o /dev/null $R/v2/ 2>&1 | grep -iE "HTTP/|www-authenticate"

echo "===== 2. anonymous token for a public repo ====="
REPO=dcu/ac47bbmz64/base/jupyterlab-pytorch
AUTH=$(timeout 10 curl -sSk -D- -o /dev/null $R/v2/ | grep -i www-authenticate | tr -d '\r')
echo "challenge: $AUTH"
REALM=$(echo "$AUTH" | sed -n 's/.*realm="\([^"]*\)".*/\1/p')
SERVICE=$(echo "$AUTH" | sed -n 's/.*service="\([^"]*\)".*/\1/p')
echo "realm=$REALM service=$SERVICE"
if [ -n "$REALM" ]; then
  TOK=$(timeout 10 curl -sSk "$REALM?service=$SERVICE&scope=repository:$REPO:pull" | python3 -c "import json,sys;print(json.load(sys.stdin).get('token',''))" 2>/dev/null)
  echo "token_len=${#TOK}"
  echo "--- manifest (v2) ---"
  timeout 10 curl -sSk -o /dev/null -w "manifest => %{http_code}\n" -H "Authorization: Bearer $TOK" \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json" \
    "$R/v2/$REPO/manifests/2.7.1-ubuntu22.04-dtk26.04-py3.11-devel_longcatnew3"
fi

echo "===== 3. sglang image path ====="
REPO2=dcu/jsyadmin/base/jupyter-sglang
TOK2=""
if [ -n "$REALM" ]; then
  TOK2=$(timeout 10 curl -sSk "$REALM?service=$SERVICE&scope=repository:$REPO2:pull" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('token','') or d.get('errors',''))" 2>/dev/null)
fi
echo "sglang token/err: ${TOK2:0:120}"
[ -n "$TOK2" ] && timeout 10 curl -sSk -o /dev/null -w "sglang manifest => %{http_code}\n" -H "Authorization: Bearer $TOK2" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.index.v1+json" \
  "$R/v2/$REPO2/manifests/jupyter-sglang-from20091"

echo "===== 4. catalog ====="
[ -n "$TOK" ] && timeout 10 curl -sSk -H "Authorization: Bearer $TOK" "$R/v2/_catalog?n=5" 2>&1 | head -c 400
echo
echo "===== DONE ====="
