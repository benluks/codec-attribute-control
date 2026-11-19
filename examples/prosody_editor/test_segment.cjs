const fs=require('fs'),assert=require('assert');
const s=fs.readFileSync(require('path').join(__dirname,'index.html'),'utf8');
const code=s.slice(s.indexOf('function voicedSegment('),s.indexOf('p.onpointerdown=e=>'));
const frames=Array.from({length:90},(_,i)=>({t:i/80,hz:(i>=20&&i<=28)||(i>=46&&i<=56)?null:140+20*Math.sin(i/7)}));
const offsets=frames.map((f,i)=>f.hz==null?0:Math.sin(i/8)*.3);
const test=new Function('origF','offsets',code+';return {voicedSegment,deform};')(frames,offsets);
const seg=test.voicedSegment(36);assert.deepStrictEqual(seg,{lo:29,hi:45});
const snapshot=offsets.slice(),anchor=frames[36].t,minT=frames[30].t,maxT=frames[42].t;
const start={t:anchor,q:{},snapshot,segment:seg,minT,maxT,leftRadius:anchor-minT,rightRadius:maxT-anchor,first:30,last:42};
for(const dx of [-1,-.2,-.04,0,.03,.2,1]){
 offsets.splice(0,offsets.length,...snapshot);
 test.deform(start,dx,1.5);
 for(let i=0;i<frames.length;i++){
  assert(Number.isFinite(offsets[i]),'nonfinite '+i);
  if(i<31||i>41)assert.strictEqual(offsets[i],snapshot[i],`changed outside support at ${i} dx ${dx}`);
 }
}
console.log('PASS: segment isolation and finite offsets for extreme drags');
