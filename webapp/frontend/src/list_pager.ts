// Capture 의 Recent Sessions 와 Calibration 의 Calibration List 가 함께 쓰는 페이저.
// 목록은 최신순으로 내려오므로 1페이지가 가장 최근 항목이다.

export function renderListPager(pager: HTMLElement | null, pageIndex: number, pageCount: number): void {
  if (!pager) return;
  if (pageCount <= 1) {
    pager.hidden = true;
    pager.innerHTML = "";
    return;
  }
  pager.hidden = false;
  pager.innerHTML = `
    <button class="button secondary" type="button" data-pager-step="-1" ${pageIndex === 0 ? "disabled" : ""}>Newer</button>
    <span class="list-pager-status">${pageIndex + 1} / ${pageCount}</span>
    <button class="button secondary" type="button" data-pager-step="1" ${pageIndex >= pageCount - 1 ? "disabled" : ""}>Older</button>
  `;
}

/** 페이저 클릭이면 이동할 페이지 수를, 아니면 0 을 돌려준다. */
export function pagerStep(target: EventTarget | null): number {
  const button = (target as HTMLElement | null)?.closest<HTMLButtonElement>("[data-pager-step]");
  if (!button || button.disabled) return 0;
  return Number(button.dataset.pagerStep) || 0;
}
