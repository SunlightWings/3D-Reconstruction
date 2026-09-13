import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

export default function GaussianViewer({ modelUrl, onReset }) {
  const mount = useRef(null); const resetView = useRef(() => {}); const [error, setError] = useState("");
  useEffect(() => {
    if (!modelUrl || !mount.current) return undefined;
    const host = mount.current; const scene = new THREE.Scene(); scene.background = new THREE.Color("#102b36");
    const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000); camera.position.set(0, 0, 3);
    const renderer = new THREE.WebGLRenderer({ antialias: true }); host.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
    resetView.current = () => { camera.position.set(0, 0, 3); controls.target.set(0, 0, 0); controls.update(); };
    const spark = new SparkRenderer({ renderer }); scene.add(spark);
    let splat;
    try { splat = new SplatMesh({ url: modelUrl }); scene.add(splat); } catch { setError("The reconstruction model could not be loaded."); }
    const resize = () => { const { width, height } = host.getBoundingClientRect(); renderer.setSize(width, height); camera.aspect = width / height; camera.updateProjectionMatrix(); };
    resize(); const observer = new ResizeObserver(resize); observer.observe(host);
    renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
    return () => { observer.disconnect(); renderer.setAnimationLoop(null); controls.dispose(); renderer.dispose(); splat?.removeFromParent(); host.replaceChildren(); };
  }, [modelUrl]);
  const fullscreen = () => mount.current?.requestFullscreen?.();
  return <><div className="viewer" ref={mount}>{error && <p>{error}</p>}</div><div className="viewer-actions"><span>Drag to rotate · Scroll or pinch to zoom · Right drag to pan</span><button onClick={() => resetView.current()}>Reset view</button><button onClick={fullscreen}>Fullscreen</button><button onClick={onReset}>Reconstruct another</button></div></>;
}
