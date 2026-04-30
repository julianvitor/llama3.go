package downloader

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type Progress struct {
	Downloaded int64
	Total      int64
	SpeedBps   float64
}

type Options struct {
	URL         string
	DestPath    string
	HTTPClient  *http.Client
	UserAgent   string
	BearerToken string
	// If non-empty, verifies the SHA256 of the final file.
	ExpectedSHA256 string
	// If true, will overwrite existing file.
	Force bool
}

func Download(ctx context.Context, opt Options, onProgress func(p Progress)) error {
	if opt.URL == "" {
		return fmt.Errorf("missing URL")
	}
	if opt.DestPath == "" {
		return fmt.Errorf("missing DestPath")
	}
	if opt.HTTPClient == nil {
		opt.HTTPClient = &http.Client{Timeout: 0}
	}
	if opt.UserAgent == "" {
		opt.UserAgent = "go-llama-chat/0.1"
	}

	if err := os.MkdirAll(filepath.Dir(opt.DestPath), 0o755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}

	// If file exists and not forcing, verify SHA (if provided) and return.
	if st, err := os.Stat(opt.DestPath); err == nil && st.Size() > 0 && !opt.Force {
		if opt.ExpectedSHA256 != "" {
			ok, err := verifySHA256(opt.DestPath, opt.ExpectedSHA256)
			if err != nil {
				return err
			}
			if ok {
				return nil
			}
			// hash mismatch, continue with re-download
		}
	}

	// Determine remote size (best effort)
	remoteSize, _ := headContentLength(ctx, opt)

	// Resume support: write to .part and then rename.
	partPath := opt.DestPath + ".part"
	var existing int64
	if st, err := os.Stat(partPath); err == nil {
		existing = st.Size()
	}
	if existing > 0 && remoteSize > 0 && existing > remoteSize {
		_ = os.Remove(partPath)
		existing = 0
	}

	f, err := os.OpenFile(partPath, os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open part: %w", err)
	}
	defer f.Close()
	if _, err := f.Seek(existing, io.SeekStart); err != nil {
		return fmt.Errorf("seek part: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, opt.URL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", opt.UserAgent)
	if opt.BearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+opt.BearerToken)
	}
	if existing > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existing))
	}

	resp, err := opt.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK && existing > 0 {
		// server ignored range; start over
		_ = os.Remove(partPath)
		existing = 0
		f.Close()
		f, err = os.OpenFile(partPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			return fmt.Errorf("reopen part: %w", err)
		}
		defer f.Close()
	}

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("download failed: %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}

	total := remoteSize
	if total <= 0 {
		if cl := resp.Header.Get("Content-Length"); cl != "" {
			if n, err := strconv.ParseInt(cl, 10, 64); err == nil {
				if resp.StatusCode == http.StatusPartialContent {
					total = existing + n
				} else {
					total = n
				}
			}
		}
	}

	buf := make([]byte, 1024*1024)
	downloaded := existing
	lastTime := time.Now()
	lastBytes := downloaded

	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := f.Write(buf[:n]); err != nil {
				return fmt.Errorf("write: %w", err)
			}
			downloaded += int64(n)
		}

		now := time.Now()
		if now.Sub(lastTime) >= 200*time.Millisecond {
			deltaBytes := downloaded - lastBytes
			speed := float64(deltaBytes) / now.Sub(lastTime).Seconds()
			onProgress(Progress{Downloaded: downloaded, Total: total, SpeedBps: speed})
			lastTime = now
			lastBytes = downloaded
		}

		if rerr != nil {
			if errors.Is(rerr, io.EOF) {
				break
			}
			return rerr
		}
	}

	// final progress emit
	onProgress(Progress{Downloaded: downloaded, Total: total, SpeedBps: 0})

	if err := f.Close(); err != nil {
		return err
	}

	if opt.ExpectedSHA256 != "" {
		ok, err := verifySHA256(partPath, opt.ExpectedSHA256)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("sha256 mismatch for %s", filepath.Base(opt.DestPath))
		}
	}

	if err := os.Rename(partPath, opt.DestPath); err != nil {
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}

func headContentLength(ctx context.Context, opt Options) (int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, opt.URL, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set("User-Agent", opt.UserAgent)
	if opt.BearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+opt.BearerToken)
	}
	resp, err := opt.HTTPClient.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return 0, fmt.Errorf("head failed: %s", resp.Status)
	}
	cl := resp.Header.Get("Content-Length")
	if cl == "" {
		return 0, fmt.Errorf("no content-length")
	}
	n, err := strconv.ParseInt(cl, 10, 64)
	if err != nil {
		return 0, err
	}
	return n, nil
}

func verifySHA256(path string, expectedHex string) (bool, error) {
	expectedHex = strings.ToLower(strings.TrimSpace(expectedHex))
	if expectedHex == "" {
		return true, nil
	}
	b, err := hex.DecodeString(expectedHex)
	if err != nil {
		return false, fmt.Errorf("invalid expected sha256: %w", err)
	}
	if len(b) != sha256.Size {
		return false, fmt.Errorf("expected sha256 must be 32 bytes")
	}

	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false, err
	}
	sum := h.Sum(nil)
	return bytesEqual(sum, b), nil
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	var v byte
	for i := 0; i < len(a); i++ {
		v |= a[i] ^ b[i]
	}
	return v == 0
}
