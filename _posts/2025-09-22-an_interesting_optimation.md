---
layout:     post
title:      "记一个有趣的编译优化选项 `-enable-dfa-jump-thread`"
subtitle:   " \"Hello World, Hello Blog\""
date:       2025-09-22 12:00:00
author:     "nothin"
header-img: "img/bg-walle.jpg"
catalog: true
tags:
    - 编译优化
---





# 记一个有趣的编译优化选项

> 前言是师姐在测试coremark时，发现gcc和icx的性能比llvm的性能要好。查看汇编代码发现，gcc和icx能将coremark中的一个状态机代码优化为使用goto串联起来状态转换过程，从而不需要使用跳转表来执行跳转流程。然后交给我任务来调研llvm中是否有相关的优化。

## coremark源代码

coremark中的代码位于https://github.com/eembc/coremark/blob/main/core_state.c，一个经典的for套switch的状态机实现。

经过笔者调研，这是一种优化算法，但是可以直接手写为优化后的结果，类似的实现见：https://dev.to/lexplt/implementing-computed-gotos-in-c-193p

该网站的示例代码：

```cpp
std::vector<uint8_t> bytecode = readBytecode();
std::size_t pos = 0;
uint8_t inst = bytecode[pos];
{
    dispatch_op:
    switch (inst)
    {
        case DO_THIS:
            // ...
            inst = bytecode[pos];
            pos++;
            goto dispatch_op;
        case DO_THAT:
            // ...
            inst = bytecode[pos];
            pos++;
            goto dispatch_op;
        // ...
    }
```



终极优化后的代码为

```cpp
#define FETCH_INSTRUCTION()    \
    do {                       \
        inst = bytecode[pos];  \
        pos++;                 \
    } while (false)

#if USE_COMPUTED_GOTO
#  define DISPATCH_GOTO() goto opcodes[inst]
#  define TARGET(op) TARGET_##op
#else
#  define DISPATCH_GOTO() goto dispatch_op
#  define TARGET(op) case op:
#end

#define DISPATCH()        \
    FETCH_INSTRUCTION();  \
    DISPATCH_GOTO()

std::vector<uint8_t> bytecode = readBytecode();
std::size_t pos = 0;
uint8_t inst = bytecode[pos];
{
#if !USE_COMPUTED_GOTO
    dispatch_op:
    switch (inst)
#else
    const std::array opcodes = {
        &&TARGET_DO_THIS,
        &&TARGET_DO_THAT,
        &&TARGET_STOP_INTERPRETER
    };
#end
    {
        TARGET(DO_THIS)
        {
            // ...
            DISPATCH();
        }
        TARGET(DO_THAT)
        {
            // ...
            DISPATCH();
        }
        TARGET(STOP_INTERPRETER)
        {
            goto label_end;
        }
        // ...
    }
    label_end:
    do {} while (false);
}
```

这两个代码编译后的运行时间可以参见网站描述。

coremark代码相关优化结果[示例代码](https://godbolt.org/#z:OYLghAFBqRAWIDGB7AJgUwKKoJYBdkAnAGhxAgDMcAbdAOwEMBbdEAcgEY3iLk68Ayoga0QHACw8%2BeAKoBndAAUAHuwAM3AFZji1BnVCIApACYAQqbPEFtRHhx9y9VAGFk1AK5M6O5wBkcOnQAOS8AI3RCMQBmAHZiAAdkOXwHOjdPbx0klPs%2BAKDQpgiojjjrdFs8ugE8BkI8DK8fDgqqtNr6vAKQ8MiY%2BLk6hqas1qGunqKSgYBKa2QPQkRWNnovAGoXAHkAJUwAfQEAFQBBY8wjNVOjWItrjcetvcOT093jjaNogBENtWIV1OT2e%2BwOAElggA1U5%2BcE/QEPJ47MECVpAkEo14mRHA5EvCHBY64zEEgBifm25xJ%2BLBmAAGoptsFMESaY8sUcXODWcdwWTwS52RtgjIALIHTlvC4CIG3H7fe43a4TeyIDaIOD1IHodAHHByXDAfAQTX1DWzOV3DFPM2EDaEdB4ABuIkVNsejpdIi%2Bvw2UHV30ufrAbDUodmXxMADZ/YHoi5vn9QwBOCOR75kjYcDYgf7upGep1LOgOp2u6gFm6xBXXHV0TZSs4XIEoR0HVV6vCEfS5NKmrX2gBUQ8CQxIG0CeA2Q%2B7vdSfAOKA8/EtdethY1g5nIN3499f1HdHHVZBdt3F5Z9OORwAmmKzNs/KenusmKDXs3MBtOweP0czg%2BF9Hl4e0IEVGd91MaNoJ/Oo8HQDYwDAJN/0hGE4QVaIzDgwhLEsNc8UeW4lQvR4rxvAR70fPw/yHE9sI9XccAof0KLvB8nwPVDQ2ICMNgAeiHDZnA2ZBWLwOADUnOgEg8achwEpiQRI5SyJPcx8MYzcyI2MJHQYABrYDd3lNTHjkAB3fBNX9TtCN01SdLI4QFH/N4PhAczdMnViIF1fVDRwY08AgdiqM4vxZgcnyLycojYvU%2BDENQzlIWOEzEq%2BGtvN0yo3JYtiGUo6iuKTHi2EsUNssTWIXBFYqOJo7iQzYABaCNcrI%2BKst039UoJNFMsSsznNi/LEMKsLGoi5rytagA6Tqxp8nreovfq/U5CkqQy7SEqy0aDsSiaurijdjvWza/jS6FYXhYb1rnY8FzoJdFn4IwAFYzFujCHq%2BhVNM0/b1uInKVt056%2B0XZdPp%2Bpt3gywH8JB0iwf09AjMe3dXMQqUOC8yGyKmgKDSNE1wtKqKYvWtawaea60KJHGsuh173pXPBvt%2BwaOG%2BoGLDRs7TIhy6fIm3yiuvJqyt%2BCqlrYWnevphm4IYBC/22ylzlZxL2eqTn4d51F%2BZR4GLFBsGjoZ07ifO9G1fVzWBrBdD7qwx21YNtIje5hG%2BYF1HLa9w6xYZzHsat2K8eZ44ifF3SpqpyKWuTNhFeVsPQ7Bpntd2vXYp92GPv9k3DnSoOLcL0Xa0TsjJam5CyaCkLppl2an2ikWVIup2LOSrWCXdzCa6hnsXsNuGy9u5HBa0nPYptjGDOM6OfNj/PzgThnk5m6m06QthMGq25avqlO5vl1r0GW%2BuHZ7pKNZSrbBpMMeyOLt7p55re5%2BDj%2B4M6622oAVPyzc9Tk2CpTfekVu721FovXqedh53VHuvBmX8/a/3JDrf%2B1cMEjXDivLGa8kFngYG5KUJgd5gz3h3A%2B8105VTYDVM%2BDUGGpyYUfDqStH7g3IYlFBdJGTMl5IA3cWCf4B1RO/c2QsQ78OysAsGdt76IKUSCYRFc0EPUIU9CeMNv6lxwbIquCiJHL3WpHMhItN4EgZEyFkRJaHrVJpA1uMDOE0Xgeo3ugjYraK5DyIk/JBQSJBFIkxMjDiOLESzeRC8lFWN6mohmqsGZBJHnogJPkolc1MbE0Rzj8EWP0atYh1jV413saibkvIwkuFcb1JuYAW4U1CpfLuWciG5L6oPV2Oj/qe00U8fJxs/oe3MUkhBvcVFVNITXDAFAGAeGoHgZpWUbE4xSSko8UE/QMS9l6Eszt0BViOkrXQ7AvrcB8GGbgyB2CJgtj%2BRYywUomGiFwYgGyHnzHgEgZATAEg0EiKQcgKAQVgqiIgPQBgTBqA4ACKg6zIhyHIGEdQ3AwiBHqAAT3YD83FjBCD4u2GETQ6A7BEu4FClg/Bth0GoISh5xAMBhA8MAFwIhQG0vZegJg%2BhgCiDZbgR0dgcDOnQBitl6BlDUvkqsH5U5KjYt0DgfSBK3AYHVd2HATB%2BXSsIGEZI6AfiCuFdQQIoB/k8D0MAOQUIcDoEstsBI9B%2BW8H4EIEQrAJBSH4PIJQqg2XaBxPCwwwcNVhAxZAeYyAEjVFlW1bYJhHnGsIDgDAcaIDzBsNS6oTg6CuHcM0XwJaph9FKIkZIr1RgtFrUYqtxR%2BjjEqIWjowxGhlrGG0TtfBOgNBbTMcY3aG06AmMOwIvRW2lHze8lYYh5jUBuXc9VTzWHKAABzRjatGcQGpI0bERQtDgC01D%2BmwPgIgUZvmzG4H8jQAKEBQtBbQEgZBTTAvff0OFwrEXIp4DQBChAMUQCxWyklBL%2BXQbJRSqlNLOB0uBQyvATKWXqo5Vynl1A%2BXIYFUKgwoqNDspwBK%2Bw0rZWkflYqhC/LVWrrZdarVZKdWrFI/qw1BHjWmoUBaojIqbXYvmBQB1TqXVuo9fcn53rBDCFEAGuTwaVDqu0NEXQwqkDRpY7mhNSa0gprTRsNqTA8POnfG1egDAwi0DaqgFZbVNBeASG1SSBlUAZsiFmnN4A839slY4fyJaJ2tH8DO6Ybam31t7Y2nIr0R1RYLYFmo47YuTo7Slod3QIvVsnWlzIjap05cKHljgC6lhLvK9ctgtziD3NI5ujYO690HqPcKk9agz0XqvbgAg9pTD3sfSJ4ggK30wohd%2B6FH6QA4EQMoRFJgOAcBxKi0D4HIOkbg6y4leL4OUsLfy%2Bl9B0PMtZaR7D3LeWyp%2BRgQTJHuDis7VR9VtHEBKoY/wNVzHNU9jY2gDjj6s3cZ%2Bbxs1AmrXCbtWJhgjrnWuvdZ6gjcnfWKckMphQqmw06Ejdpi2Ma9PEETcm9gqafnIEzdmmVfn82ZaLcF0thWK2oESzW%2BL1RQvReqKzjL7RB0FfLe2vnqXJi5bnflronPis8%2Bq3IRd/qV1rvqxu9gzXd37sPXN5QnWlvdZML1m9A2vnleG/8%2BYhkxBqABEx8Q3BDXIoBA1x57BuAYoBE%2BpW8xjUpEcOIIAA%3D%3D)，

## 优化选项

经过笔者的调研，这个有趣的优化已经在llvm中实现了，选项为`-enable-dfa-jump-thread`，相应的commit链接为https://github.com/llvm/llvm-project/commit/02077da，不过默认关闭。

经过笔者测试，在O2基础上关闭和打开这个选项，会带来一定的性能提升。

* 硬件设置  macOS m4处理器，16GB内存。
* clang版本 ` clang version 14.0.6` 。

使用不同选项下，测试coremark的iteractions/sec为

| -O2                       | -O2 -mllvm -enable-dfa-jump-thread |
| ------------------------- | ---------------------------------- |
| 600000/12.211=`49136.024` | 1100000/19.807=`55,535.922`        |

可见，能够获得比较明显的性能提升。

## 附

除此之外，笔者还学到一个新的语法，使用  `&&`  运算符，获得一个label的地址，存储到一个数组中。然后goto 语句的目标地址使用索引来计算目标地址。见https://gcc.gnu.org/onlinedocs/gcc/Labels-as-Values.html