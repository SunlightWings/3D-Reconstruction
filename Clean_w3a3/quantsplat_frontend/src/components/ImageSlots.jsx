import { useEffect, useRef, useState } from "react";

function PhotoSlot({ file, index, disabled, onSelect, onRemove }) {
  const [preview, setPreview] = useState("");
  useEffect(() => {
    if (!file) { setPreview(""); return undefined; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return <article className={`slot ${file ? "complete" : ""}`}>
    <button className="slot-image" type="button" onClick={() => onSelect(index)} disabled={disabled}>
      {preview ? <img src={preview} alt={`Selected view ${index + 1}`} /> : <span>＋<small>Photo {index + 1}</small></span>}
    </button>
    <div className="slot-meta">
      <span title={file?.name}>{file ? file.name : "Add a view"}</span>
      {file && <span className="slot-actions"><button type="button" className="text-button" onClick={() => onSelect(index)} disabled={disabled}>Replace</button><button type="button" className="text-button" onClick={() => onRemove(index)} disabled={disabled}>Remove</button></span>}
    </div>
  </article>;
}

export default function ImageSlots({ files, onChange, disabled }) {
  const input = useRef(null);
  const select = (index) => {
    if (disabled) return;
    input.current.dataset.index = index;
    input.current.click();
  };
  const receive = (event) => {
    const index = Number(event.currentTarget.dataset.index);
    const next = [...files];
    next[index] = event.target.files?.[0] || null;
    onChange(next);
    event.target.value = "";
  };
  const remove = (index) => onChange(files.map((file, i) => i === index ? null : file));

  return <>
    <input ref={input} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={receive} />
    <div className="slots">
      {files.map((file, index) => <PhotoSlot key={index} file={file} index={index} disabled={disabled} onSelect={select} onRemove={remove} />)}
    </div>
  </>;
}
