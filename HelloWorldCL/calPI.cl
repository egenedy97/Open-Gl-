//root ( 1- square(i/n) )
// the value get added to the output

kernel void calPI(global float* in1, global float* in2, global float* out)
{
    size_t i = get_global_id(0);
    out[i]=sqrt(1-pow((in1[i]/in2[i]),2));
}
