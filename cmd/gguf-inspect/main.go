package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"go_llama/internal/llama"
)

func main() {
	path := flag.String("path", os.ExpandEnv("$HOME/.cache/go-llama/Llama-3.2-1B-Instruct-Q8_0.gguf"), "Path to .gguf")
	filter := flag.String("filter", "", "Substring filter")
	findTok := flag.String("find-token", "", "Search in tokenizer.ggml.tokens (substring)")
	listTok := flag.Bool("list-tokenizer", false, "List first N tokenizer tokens")
	listTokN := flag.Int("n", 50, "N for --list-tokenizer")
	listKV := flag.Bool("list-kv", false, "List KV keys")
	kvFilter := flag.String("kv-filter", "", "Substring filter for --list-kv")
	getKV := flag.String("get-kv", "", "Print a KV value by exact key")
	flag.Parse()

	f, err := llama.OpenGGUF(*path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer f.Close()

	if *getKV != "" {
		kv, ok := f.GGUF.KV[*getKV]
		if !ok {
			fmt.Fprintln(os.Stderr, "missing key")
			os.Exit(1)
		}
		fmt.Printf("%s\t(type=%d)\n", kv.Key, kv.Type)
		fmt.Printf("%#v\n", kv.Value)
		return
	}

	if *listKV {
		keys := make([]string, 0, len(f.GGUF.KV))
		for k := range f.GGUF.KV {
			if *kvFilter != "" && !strings.Contains(k, *kvFilter) {
				continue
			}
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			kv := f.GGUF.KV[k]
			fmt.Printf("%s\t(type=%d)\n", kv.Key, kv.Type)
		}
		return
	}

	if *findTok != "" || *listTok {
		g := f.GGUF
		v, ok := g.GetStrings("tokenizer.ggml.tokens")
		if !ok {
			fmt.Fprintln(os.Stderr, "missing tokenizer.ggml.tokens")
			os.Exit(1)
		}
		if *listTok {
			n := *listTokN
			if n > len(v) {
				n = len(v)
			}
			for i := 0; i < n; i++ {
				fmt.Printf("%d\t%q\n", i, v[i])
			}
			return
		}
		for i, s := range v {
			if strings.Contains(s, *findTok) {
				fmt.Printf("%d\t%q\n", i, s)
			}
		}
		return
	}

	keys := make([]string, 0, len(f.ByName))
	for k := range f.ByName {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		if *filter != "" && !strings.Contains(k, *filter) {
			continue
		}
		t := f.ByName[k]
		fmt.Printf("%s\t(type=%d dims=%v)\n", t.Info.Name, t.Info.Type, t.Info.Dims)
	}
}
