/**
 * A model's flow chart, as a tree of blocks.
 *
 * `id` is unique within the chart. `target` is the page section a click jumps to; it
 * defaults to `id`, and differs only when one block appears more than once in the
 * flow (both LayerNorms in a transformer block jump to the one LayerNorm section).
 */
export interface FlowNode {
  id: string;
  label: string;
  target?: string;
  shape?: string;
  repeat?: string;
  children?: FlowNode[];
}

export interface BreadcrumbStep {
  label: string;
  target: string;
}

export function targetOf(node: FlowNode): string {
  return node.target ?? node.id;
}

export function breadcrumbTo(root: FlowNode, target: string): BreadcrumbStep[] {
  const path = findPath(root, target);
  return (path ?? []).map((node) => ({ label: node.label, target: targetOf(node) }));
}

function findPath(node: FlowNode, target: string): FlowNode[] | null {
  if (targetOf(node) === target) return [node];
  for (const child of node.children ?? []) {
    const path = findPath(child, target);
    if (path) return [node, ...path];
  }
  return null;
}
